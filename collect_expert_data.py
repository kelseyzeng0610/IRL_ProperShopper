"""
Collect expert demonstrations 


 Run this script: python collect_expert_data.py --num_episodes 100
"""

import socket
import json
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_recorder import TrajectoryRecorder
from utils import recv_socket_data

# Import expert functions
from socket_agent import (
    get_container,
    put_food_in_container,
    checkout,
    exit_supermarket,
    step
)


def collect_demonstrations(num_episodes=100, save_dir="./demonstrations", host='127.0.0.1', port=9000):
    """
    Run expert agent and collect demonstrations

    Returns:
        TrajectoryRecorder with collected trajectories
    """
    recorder = TrajectoryRecorder(save_dir=save_dir)

    print(f"Connecting to environment at {host}:{port}...")
    sock_game = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock_game.connect((host, port))
        print("✓ Connected to environment")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nMake sure to start the environment first:")
        print("  python socket_env.py")
        return None

    print(f"\nCollecting {num_episodes} expert demonstrations...")

    successes = 0
    failures = 0
    skipped_violations = 0

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        try:
            # Reset environment
            output = step(sock_game, "RESET")

            if output is None or output == b'':
                print("  Environment closed")
                break

            shopping_list = output['observation']['players'][0]['shopping_list']
            list_quant = output['observation']['players'][0]['list_quant']
            print(f"  Shopping list: {list(zip(shopping_list, list_quant))}")

            # Start recording
            recorder.start_episode(output)

            # Execute expert policy and check for violations
            timestep = 0
            has_violations = False
            
            def step_and_record(sock, action_str):
                """Wrapper to step and record"""
                nonlocal timestep
                result = step(sock, action_str)
                recorder.record_step(action_str, result, timestep)
                timestep += 1
                return result

            # Temporarily replace step function
            import socket_agent
            original_step = socket_agent.step
            socket_agent.step = step_and_record

            try:
                # Step 1: Get container (cart or basket)
                print("  → Getting container...")
                output = get_container(output, sock_game)
                
                # Check for violations
                if output and 'violations' in output and output['violations'] and output['violations'] != '':
                    print(f"  Violation detected: {output['violations']}")
                    has_violations = True

                # Step 2: Collect each item
                if not has_violations:
                    for i, food in enumerate(shopping_list):
                        qty = list_quant[i]
                        print(f"  → Collecting {qty}x {food}...")

                        # Collect each quantity separately
                        for _ in range(qty):
                            output = put_food_in_container(output, sock_game, food)
                            
                            # Check for violations
                            if output and 'violations' in output and output['violations'] and output['violations'] != '':
                                print(f" Violation detected: {output['violations']}")
                                has_violations = True
                                break
                        
                        if has_violations:
                            break

                # Step 3: Checkout
                if not has_violations:
                    print("  → Checking out...")
                    output = checkout(output, sock_game)
                    
                    # Check for violations
                    if output and 'violations' in output and output['violations'] and output['violations'] != '':
                        print(f"  Violation detected: {output['violations']}")
                        has_violations = True

                # Step 4: Exit
                if not has_violations:
                    print("  → Exiting store...")
                    output = exit_supermarket(output, sock_game)
                    
                    # Check for violations
                    if output and 'violations' in output and output['violations'] and output['violations'] != '':
                        print(f" Violation detected: {output['violations']}")
                        has_violations = True
                        
            finally:
                # Restore original step function
                socket_agent.step = original_step

            # Only save if no violations occurred
            if has_violations:
                print(f"  Episode {episode + 1} skipped due to violations")
                recorder.end_episode(success=False, final_obs=output)
                skipped_violations += 1
                continue
            
            # Episode succeeded without violations
            recorder.end_episode(success=True, final_obs=output)
            successes += 1
            print(f"  Episode {episode + 1} completed successfully ({timestep} steps, no violations)")

        except RuntimeError as e:
            # Expert policy timeout/failure
            print(f"  Episode {episode + 1} failed: {e}")
            recorder.end_episode(success=False)
            failures += 1
            continue

        except Exception as e:
            # Other errors
            print(f"  Episode {episode + 1} error: {e}")
            recorder.end_episode(success=False)
            failures += 1
            continue

    # Close socket
    sock_game.close()

    # Print and save statistics
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Episodes attempted: {num_episodes}")
    print(f"Successes (no violations): {successes}")
    print(f"Skipped (violations): {skipped_violations}")
    print(f"Failures (other errors): {failures}")
    print(f"Success rate: {successes/num_episodes*100:.1f}%")

    if successes > 0:
        recorder.print_statistics()
        filename = recorder.save()
        print(f"\n✓ Demonstrations saved to: {filename}")
    else:
        print("\n✗ No successful episodes to save")

    return recorder


def main():
    parser = argparse.ArgumentParser(description='Collect expert demonstrations')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--save_dir', type=str, default='./demonstrations',
                       help='Directory to save demonstrations')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Socket host')
    parser.add_argument('--port', type=int, default=9000,
                       help='Socket port')

    args = parser.parse_args()

    print("="*60)
    print("EXPERT DEMONSTRATION COLLECTION")
    print("="*60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Save directory: {args.save_dir}")
    print(f"Host: {args.host}:{args.port}")
    print()
    print("NOTE: Make sure the environment is running first:")
    print("  python socket_env.py")

    recorder = collect_demonstrations(
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        host=args.host,
        port=args.port
    )

    if recorder and len(recorder.all_trajectories) > 0:
        print("\n Data collection complete!")
        print(f"\nNext steps:")
        print(f"  1. Check the summary file in {args.save_dir}")
        print(f"  2. Train BC model: python train_bc.py --data_path <trajectory_file>")
    else:
        print("\n✗ Data collection failed")
        print("  Make sure socket_env.py is running")


if __name__ == "__main__":
    main()
