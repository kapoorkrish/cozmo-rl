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
- Left wheel velocity [-1, 1]
- Right wheel velocity [-1, 1]
- Lift height [32, 92]
- Height angle [-0.44, 0.78]

# Observation Space (8)
### Translational Position
- pose_x
- pose_y
### Angular Position
- sin(pose_angle_rad)
- cos(pose_angle_rad)
### Translational Velocity
- rwheel_speed_mmps
- lwheel_speed_mmps
### Lift Height
- lift_height_mm
### Head Angle
- head_angle_rad