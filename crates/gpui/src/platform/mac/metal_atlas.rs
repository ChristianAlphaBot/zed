use crate::{
    AtlasEntry, AtlasKey, AtlasSweep, AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds,
    DevicePixels, PlatformAtlas, Point, Scene, Size, platform::AtlasTextureList,
};
use anyhow::{Context as _, Result};
use collections::{FxHashMap, FxHashSet};
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use metal::Device;
use parking_lot::Mutex;
use std::borrow::Cow;

pub(crate) struct MetalAtlas(Mutex<MetalAtlasState>);

impl MetalAtlas {
    pub(crate) fn new(device: Device) -> Self {
        MetalAtlas(Mutex::new(MetalAtlasState {
            device: AssertSend(device),
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            tiles_by_key: Default::default(),
            sweep: AtlasSweep::new(AtlasSweep::EVERY),
        }))
    }

    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> metal::Texture {
        self.0.lock().texture(id).metal_texture.clone()
    }
}

struct MetalAtlasState {
    device: AssertSend<Device>,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasEntry>,
    /// cipher patch: when it lets go of the glyphs and SVGs no frame draws any more.
    sweep: AtlasSweep,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        let state = &mut *lock;
        if let Some(entry) = state.tiles_by_key.get_mut(key) {
            state.sweep.touch(entry);
            return Ok(Some(entry.tile.clone()));
        }
        let Some((size, bytes)) = build()? else {
            return Ok(None);
        };
        let tile = state
            .allocate(size, key.texture_kind())
            .context("failed to allocate")?;
        let texture = state.texture(tile.texture_id);
        texture.upload(tile.bounds, &bytes);
        let entry = state.sweep.entry(tile.clone());
        state.tiles_by_key.insert(key.clone(), entry);
        Ok(Some(tile))
    }

    fn usage(&self) -> crate::AtlasUsage {
        let lock = self.0.lock();
        let mut usage = crate::AtlasUsage {
            tiles: lock.tiles_by_key.len(),
            ..Default::default()
        };
        for texture in lock
            .monochrome_textures
            .textures
            .iter()
            .chain(lock.polychrome_textures.textures.iter())
            .flatten()
        {
            usage.textures += 1;
            usage.bytes += texture.metal_texture.width()
                * texture.metal_texture.height()
                * u64::from(texture.bytes_per_pixel());
        }
        usage
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();
        let Some(id) = lock.tiles_by_key.get(key).map(|v| v.tile.texture_id) else {
            return;
        };

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut lock.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut lock.polychrome_textures,
        };

        let Some(texture_slot) = textures
            .textures
            .iter_mut()
            .find(|texture| texture.as_ref().is_some_and(|v| v.id == id))
        else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.decrement_ref_count();

            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
                lock.tiles_by_key.remove(key);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }

    fn sweep(&self, scene: &Scene) {
        let mut lock = self.0.lock();
        let state = &mut *lock;
        let freed = state
            .sweep
            .sweep(std::time::Instant::now(), &mut state.tiles_by_key, scene);
        if freed.is_empty() {
            return;
        }
        for tile in &freed {
            state.release(tile);
        }
        // A texture is let go only once no key points into it, never on its count alone: upstream's
        // `remove` counts an image out but keeps its key while the texture is still counted, so a
        // count can read zero while that image, in the frame about to be presented, is still drawn
        // from the texture. A count above zero always has a key behind it, so only a texture
        // counted down to zero is looked at.
        let mut emptied: Vec<AtlasTextureId> = Vec::new();
        for tile in &freed {
            let id = tile.texture_id;
            let counted_out = state
                .textures(id.kind)
                .textures
                .get(id.index as usize)
                .and_then(Option::as_ref)
                .is_some_and(|texture| texture.live_atlas_keys == 0);
            if counted_out && !emptied.contains(&id) {
                emptied.push(id);
            }
        }
        if emptied.is_empty() {
            return;
        }
        let held: FxHashSet<AtlasTextureId> = state
            .tiles_by_key
            .values()
            .map(|entry| entry.tile.texture_id)
            .collect();
        for id in emptied {
            if held.contains(&id) {
                continue;
            }
            let textures = state.textures_mut(id.kind);
            if let Some(slot) = textures.textures.get_mut(id.index as usize) {
                if slot.take().is_some() {
                    textures.free_list.push(id.index as usize);
                }
            }
        }
    }
}

impl MetalAtlasState {
    /// cipher patch: `tile`'s place in its texture given back, and the texture counted down. The
    /// sweep lets the texture itself go, once no key points into it.
    fn release(&mut self, tile: &AtlasTile) {
        let id = tile.texture_id;
        let Some(Some(texture)) = self
            .textures_mut(id.kind)
            .textures
            .get_mut(id.index as usize)
        else {
            return;
        };
        texture.allocator.deallocate(tile.tile_id.into());
        // Upstream's `remove` may already have counted an image out of this texture: never below
        // zero.
        texture.live_atlas_keys = texture.live_atlas_keys.saturating_sub(1);
    }

    /// cipher patch: the textures of one kind.
    fn textures(&self, kind: AtlasTextureKind) -> &AtlasTextureList<MetalAtlasTexture> {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
        }
    }

    /// cipher patch: the textures of one kind, to change.
    fn textures_mut(&mut self, kind: AtlasTextureKind) -> &mut AtlasTextureList<MetalAtlasTexture> {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        }
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.into());
        texture_descriptor.set_height(size.height.into());
        let pixel_format;
        let usage;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = metal::MTLPixelFormat::A8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
        }
        texture_descriptor.set_pixel_format(pixel_format);
        texture_descriptor.set_usage(usage);
        let metal_texture = self.device.new_texture(&texture_descriptor);

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
        };

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size.into()),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> &MetalAtlasTexture {
        let textures = match id.kind {
            crate::AtlasTextureKind::Monochrome => &self.monochrome_textures,
            crate::AtlasTextureKind::Polychrome => &self.polychrome_textures,
        };
        textures[id.index as usize].as_ref().unwrap()
    }
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    live_atlas_keys: u32,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size.into())?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: allocation.rectangle.min.into(),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

impl From<Size<DevicePixels>> for etagere::Size {
    fn from(size: Size<DevicePixels>) -> Self {
        etagere::Size::new(size.width.into(), size.height.into())
    }
}

impl From<etagere::Point> for Point<DevicePixels> {
    fn from(value: etagere::Point) -> Self {
        Point {
            x: DevicePixels::from(value.x),
            y: DevicePixels::from(value.y),
        }
    }
}

impl From<etagere::Size> for Size<DevicePixels> {
    fn from(size: etagere::Size) -> Self {
        Size {
            width: DevicePixels::from(size.width),
            height: DevicePixels::from(size.height),
        }
    }
}

impl From<etagere::Rectangle> for Bounds<DevicePixels> {
    fn from(rectangle: etagere::Rectangle) -> Self {
        Bounds {
            origin: rectangle.min.into(),
            size: rectangle.size().into(),
        }
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

// cipher patch: the sweep lets a texture go only once no key points into it.
#[cfg(test)]
mod sweep_tests {
    use super::*;
    use crate::{
        FontId, GlyphId, ImageId, PolychromeSprite, RenderGlyphParams, RenderImageParams, px, size,
    };
    use std::time::Duration;

    #[test]
    fn a_texture_an_image_is_still_drawn_from_stays_when_its_count_runs_out() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let atlas = MetalAtlas::new(device);
        atlas.0.lock().sweep = AtlasSweep::new(Duration::ZERO);
        let image = AtlasKey::Image(RenderImageParams {
            image_id: ImageId(1),
            frame_index: 0,
        });
        let emoji = AtlasKey::Glyph(RenderGlyphParams {
            font_id: FontId(0),
            glyph_id: GlyphId(1),
            font_size: px(13.),
            subpixel_variant: Point::default(),
            scale_factor: 2.,
            is_emoji: true,
        });
        let insert = |key: &AtlasKey| {
            let bytes = vec![0_u8; 4 * 4 * 4];
            atlas
                .get_or_insert_with(key, &mut || {
                    Ok(Some((
                        size(DevicePixels(4), DevicePixels(4)),
                        Cow::Owned(bytes.clone()),
                    )))
                })
                .unwrap()
                .unwrap()
        };
        let drawn = insert(&image);
        assert_eq!(drawn.texture_id, insert(&emoji).texture_id, "one texture");
        // Upstream's `remove` counts the image out and keeps its key: the emoji still counts.
        atlas.remove(&image);
        let mut frame = Scene::default();
        frame.polychrome_sprites.push(PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: false,
            opacity: 1.,
            bounds: Default::default(),
            content_mask: Default::default(),
            corner_radii: Default::default(),
            tile: drawn.clone(),
        });
        // The clock starts, then two sweeps without a paint of the emoji let it go, and the
        // texture's count reaches zero while the frame still draws the image from it.
        for _ in 0..3 {
            atlas.sweep(&frame);
        }
        assert!(!atlas.0.lock().tiles_by_key.contains_key(&emoji));
        assert!(atlas.0.lock().tiles_by_key.contains_key(&image));
        // What the frame's present reads: a texture let go here was an unwrap of nothing.
        assert_eq!(atlas.metal_texture(drawn.texture_id).width(), 1024);
    }
}
