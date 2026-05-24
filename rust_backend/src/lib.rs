use pyo3::prelude::*;
use std::sync::Arc;
use twgame::core::twsnap::enums::{Direction, HookState};
use twgame::core::twsnap::flags::JumpFlags;
use twgame::core::twsnap::items::Tee;
use twgame::core::twsnap::time::Instant;
use twgame::core::twsnap::{vec2_from_bits, Velocity};
use twgame::{coord, Map};
use twmap::ndarray::Axis;
use twmap::vek::Vec2;
use twmap::TwMap;

// Tile type discriminants (from twgame::map::Tile, private module)
const TILE_KILL: u8 = 2;
const TILE_FINISH: u8 = 34;

// DDNet default tuning values
const GROUND_JUMP_IMPULSE: f32 = 13.2;
const AIR_JUMP_IMPULSE: f32 = 12.0;
const HOOK_LENGTH: f32 = 380.0;

/// Lightweight DDNet simulation environment using twgame Map physics.
#[pyclass]
pub struct LiteEnv {
    map: Arc<Map>,
    tee: Tee,
    tick_num: u64,
    prev_jump_pressed: bool,
    prev_hook_pressed: bool,
    jump_count: i32,
    max_jumps: i32,
}

#[pymethods]
impl LiteEnv {
    #[new]
    fn new(map_path: String) -> PyResult<Self> {
        let map_data = std::fs::read(&map_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read map: {}", e))
        })?;
        let mut twmap = TwMap::parse(&map_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to parse map: {}",
                e
            ))
        })?;

        let map: Map = Map::try_from(&mut twmap).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to convert map: {}",
                e
            ))
        })?;

        let map = Arc::new(map);

        let spawn = map.spawn_points[0]
            .first()
            .map(|p| coord::to_float(*p) + Vec2::new(16.0, 16.0))
            .unwrap_or(Vec2::new(160.0, 160.0));

        let pos = vec2_from_bits(spawn);

        let tee = Tee {
            tick: Instant::zero(),
            pos,
            vel: Velocity::zero(),
            direction: Direction::None,
            jumps: 2,
            ..Default::default()
        };

        Ok(LiteEnv {
            map,
            tee,
            tick_num: 0,
            prev_jump_pressed: false,
            prev_hook_pressed: false,
            jump_count: 2,
            max_jumps: 2,
        })
    }

    /// Reset player to spawn position.
    fn reset(&mut self) {
        let spawn = self.map.spawn_points[0]
            .first()
            .map(|p| coord::to_float(*p) + Vec2::new(16.0, 16.0))
            .unwrap_or(Vec2::new(160.0, 160.0));

        self.tee.pos = vec2_from_bits(spawn);
        self.tee.vel = Velocity::zero();
        self.tee.direction = Direction::None;
        self.tee.hook_state = HookState::Retracted;
        self.tee.jumped = JumpFlags::empty();
        self.tee.jumps = 2;
        self.tick_num = 0;
        self.prev_jump_pressed = false;
        self.prev_hook_pressed = false;
        self.jump_count = 2;
        self.max_jumps = 2;
    }

    /// Step one tick.
    /// Returns (pos_x, pos_y, vel_x, vel_y, grounded, dead, finished).
    #[pyo3(signature = (direction, jump, hook, aim_x, aim_y))]
    fn step(
        &mut self,
        direction: i32,
        jump: bool,
        hook: bool,
        aim_x: f32,
        aim_y: f32,
    ) -> (f32, f32, f32, f32, bool, bool, bool) {
        let pos_before =
            Vec2::new(self.tee.pos.x.to_bits() as f32, self.tee.pos.y.to_bits() as f32);
        let was_grounded = self.map.is_grounded(pos_before);

        // Direction
        self.tee.direction = match direction {
            -1 => Direction::Left,
            1 => Direction::Right,
            _ => Direction::None,
        };

        // Jump handling (edge-triggered)
        let mut vel_before = Vec2::new(
            self.tee.vel.x.to_bits() as f32 / 256.0,
            self.tee.vel.y.to_bits() as f32 / 256.0,
        );

        if jump && !self.prev_jump_pressed {
            if was_grounded {
                vel_before.y = -GROUND_JUMP_IMPULSE;
                self.jump_count = self.max_jumps;
                self.tee.jumped.set(JumpFlags::USED_JUMP_INPUT, true);
            } else if self.jump_count > 0 {
                self.jump_count -= 1;
                vel_before.y = -AIR_JUMP_IMPULSE;
                self.tee.jumped.set(JumpFlags::USED_JUMP_INPUT, true);
                if self.jump_count == 0 {
                    self.tee.jumped.insert(JumpFlags::ALL_AIR_JUMPS_USED);
                }
            }
        }
        self.prev_jump_pressed = jump;

        self.tee.vel = vec2_from_bits((vel_before * 256.0).round());

        // Hook handling (edge-triggered fire, continuous release)
        let hook_is_idle = matches!(
            self.tee.hook_state,
            HookState::Retracted | HookState::Idle
        );

        if hook && !self.prev_hook_pressed && hook_is_idle {
            self.fire_hook(aim_x, aim_y);
        } else if !hook && self.tee.hook_state == HookState::Grabbed {
            // Release hook
            self.tee.hook_state = HookState::Retracted;
        }
        self.prev_hook_pressed = hook;

        // Run physics tick
        self.tick_num += 1;
        self.map.dead_reckoning_tick(&mut self.tee);

        let pos_after =
            Vec2::new(self.tee.pos.x.to_bits() as f32, self.tee.pos.y.to_bits() as f32);
        let vel_after = Vec2::new(
            self.tee.vel.x.to_bits() as f32 / 256.0,
            self.tee.vel.y.to_bits() as f32 / 256.0,
        );

        let grounded = self.map.is_grounded(pos_after);
        if grounded {
            self.jump_count = self.max_jumps;
            self.tee.jumped = JumpFlags::empty();
        }

        // Check death: out of map
        let pos_int = coord::to_int(pos_after);
        let mut dead = self.map.is_out_of_map(pos_int);

        // Check death: kill tiles in game layer
        if !dead {
            let w = self.map.game_layer.len_of(Axis(1)) as i32;
            let h = self.map.game_layer.len_of(Axis(0)) as i32;
            let cx = pos_int.x.max(0).min(w - 1) as usize;
            let cy = pos_int.y.max(0).min(h - 1) as usize;
            dead = self.map.game_layer[[cy, cx]] as u8 == TILE_KILL;
        }

        // Check finish (game layer + front layer)
        let w = self.map.game_layer.len_of(Axis(1)) as i32;
        let h = self.map.game_layer.len_of(Axis(0)) as i32;
        let cx = pos_int.x.max(0).min(w - 1) as usize;
        let cy = pos_int.y.max(0).min(h - 1) as usize;
        let finished = self.map.game_layer[[cy, cx]] as u8 == TILE_FINISH
            || self
                .map
                .front_layer
                .as_ref()
                .map(|front| {
                    let w = front.len_of(Axis(1)) as i32;
                    let h = front.len_of(Axis(0)) as i32;
                    let cx = pos_int.x.max(0).min(w - 1) as usize;
                    let cy = pos_int.y.max(0).min(h - 1) as usize;
                    front[[cy, cx]] as u8 == TILE_FINISH
                })
                .unwrap_or(false);

        (pos_after.x, pos_after.y, vel_after.x, vel_after.y, grounded, dead, finished)
    }

    /// Get current position as (x, y) in world coordinates.
    #[getter]
    fn pos(&self) -> (f32, f32) {
        (
            self.tee.pos.x.to_bits() as f32,
            self.tee.pos.y.to_bits() as f32,
        )
    }

    /// Get current velocity as (vx, vy).
    #[getter]
    fn vel(&self) -> (f32, f32) {
        (
            self.tee.vel.x.to_bits() as f32 / 256.0,
            self.tee.vel.y.to_bits() as f32 / 256.0,
        )
    }

    /// Get map dimensions as (width_tiles, height_tiles).
    #[getter]
    fn map_size(&self) -> (usize, usize) {
        (
            self.map.game_layer.len_of(Axis(1)),
            self.map.game_layer.len_of(Axis(0)),
        )
    }

    /// Get the raw tile type at grid position (x, y).
    fn tile_at(&self, x: i32, y: i32) -> u8 {
        let w = self.map.game_layer.len_of(Axis(1)) as i32;
        let h = self.map.game_layer.len_of(Axis(0)) as i32;
        let cx = x.max(0).min(w - 1) as usize;
        let cy = y.max(0).min(h - 1) as usize;
        self.map.game_layer[[cy, cx]] as u8
    }

    /// Get a square tile grid around the player.
    /// Returns Vec<u8> of size (2*radius+1)^2 in row-major order.
    fn tile_grid(&self, radius: i32) -> Vec<u8> {
        let pos = Vec2::new(
            self.tee.pos.x.to_bits() as f32,
            self.tee.pos.y.to_bits() as f32,
        );
        let center = coord::to_int(pos);
        let w = self.map.game_layer.len_of(Axis(1)) as i32;
        let h = self.map.game_layer.len_of(Axis(0)) as i32;
        let size = (2 * radius + 1) as usize;
        let mut grid = vec![0u8; size * size];
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let tx = (center.x + dx).max(0).min(w - 1) as usize;
                let ty = (center.y + dy).max(0).min(h - 1) as usize;
                grid[((dy + radius) * (2 * radius + 1) + (dx + radius)) as usize] =
                    self.map.game_layer[[ty, tx]] as u8;
            }
        }
        grid
    }

    /// Get spawn position or None.
    fn spawn_pos(&self) -> Option<(f32, f32)> {
        self.map.spawn_points[0].first().map(|p| {
            let f = coord::to_float(*p) + Vec2::new(16.0, 16.0);
            (f.x, f.y)
        })
    }

    /// Check if a point is on solid ground.
    fn is_grounded_at(&self, x: f32, y: f32) -> bool {
        self.map.is_grounded(Vec2::new(x, y))
    }

    /// Check if a point collides with solid tiles.
    fn is_solid_at(&self, x: f32, y: f32) -> bool {
        self.map.is_solid_tee(Vec2::new(x, y))
    }

    /// Get the hook state: 0=retracted/idle, 1=flying, 2=grabbed
    #[getter]
    fn hook_state(&self) -> i32 {
        match self.tee.hook_state {
            HookState::Flying => 1,
            HookState::Grabbed => 2,
            _ => 0,
        }
    }

    /// Get jumps remaining.
    #[getter]
    fn jumps_remaining(&self) -> i32 {
        self.jump_count
    }
}

// Internal helper methods
impl LiteEnv {
    /// Fire hook in the direction of aim_x/aim_y (world-space direction).
    fn fire_hook(&mut self, aim_x: f32, aim_y: f32) {
        let tee_pos = Vec2::new(
            self.tee.pos.x.to_bits() as f32,
            self.tee.pos.y.to_bits() as f32,
        );

        let aim_dir = Vec2::new(aim_x, aim_y);
        if aim_dir.magnitude_squared() < 0.001 {
            return;
        }

        let aim_dir = twgame::core::normalize(aim_dir);
        let mut to = tee_pos + aim_dir * HOOK_LENGTH;

        let hit = self.map.intersect_hook(tee_pos, &mut to);
        if hit.is_some() {
            self.tee.hook_state = HookState::Grabbed;
            self.tee.hook_pos = vec2_from_bits(to);
            let hook_dir = to - tee_pos;
            if hook_dir.magnitude_squared() > 0.001 {
                self.tee.hook_direction =
                    vec2_from_bits((twgame::core::normalize(hook_dir) * 256.0).round());
            }
        }
    }
}

/// Python module for DDNet RL environment.
#[pymodule]
fn ddnet_rl_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LiteEnv>()?;
    Ok(())
}
