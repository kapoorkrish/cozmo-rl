# Training Pipeline
## Before MAML
- Teleoperation
- Behavioral cloning (~10 teleop examples)
- PPO fine-tuning

Foundational task is learned, with imitation learning + PPO

## After MAML
- Behavioral cloning (~1 teleop example)

New task is now learned, without PPO

# Action Space (4)
- Left wheel velocity
  - Real: [-200, 200] mm/s
  - Sim: [-15.22, 15.22] rad/s

- Right wheel velocity
  - Real: [-200, 200] mm/s
  - Sim: [-15.22, 15.22] rad/s

- Lift angle [-0.2, 0.79] rad

- Head angle [-0.44, 0.78] rad

# Observation Space (8)
### Translational Position (mm)
- pose_x
- pose_y
### Angular Position (rad)
- pose_angle_rad
### Translational Velocity (mm/s)
- rwheel_speed_mmps (mm/s)
- lwheel_speed_mmps (mm/s)
### Lift Position (rad)
- lift_height_mm
  - This should be converted to lift angle (rad)
### Head Angle (rad)
- head_angle_rad
### Vision (pixels)
- 320 x 240 x 1 image
  - Downsample to 80 x 60 x 1
  - Stack of 3 for temporal info