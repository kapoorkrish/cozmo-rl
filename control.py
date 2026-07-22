import time
import pycozmo

def main():
    with pycozmo.connect() as cli:
        print("Connected. Enabling motors...")

        # Move head up
        print("Raising head...")
        cli.set_head_angle(angle=0.6)
        time.sleep(1.5)

        # Raise lift up
        print("Raising lift...")
        cli.set_lift_height(height=92.0)
        time.sleep(1.5)

        # Drive forward for 2 seconds
        print("Driving forward...")
        cli.drive_wheels(lwheel_speed=50.0, rwheel_speed=50.0, lwheel_acc=200.0, rwheel_acc=200.0, duration=2.0)
        time.sleep(2.2)  # give it time to finish before sending the next command

        # Turn in place
        print("Turning...")
        cli.drive_wheels(lwheel_speed=-30.0, rwheel_speed=30.0, duration=1.5)
        time.sleep(1.7)
        
        # Lower lift and head back to neutral
        print("Lowering lift and head...")
        cli.set_lift_height(height=32.0)
        cli.set_head_angle(angle=0.0)
        time.sleep(1.5)

        # Stop
        cli.stop_all_motors()
        print("Done.")


if __name__ == "__main__":
    main()