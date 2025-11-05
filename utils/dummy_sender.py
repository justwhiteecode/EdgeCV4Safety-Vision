#!/usr/bin/python3
import socket
import struct
import sys
import time

# ==============================================================================
# UDP Dummy Sender for EdgeCV4Safety
# ==============================================================================
# ROLE:
# This script is used to TEST and DEBUG the robot controllers
# (e.g. SpeedControllerUDP.py) without the need to run the vision system.
#
# FUNCTIONALITY:
# The script asks the user to enter a distance value from the terminal,
# packs it in the correct format and sends it via UDP to the listener.
# ==============================================================================

# --- Destination Configuration ---
# These values MUST match the LISTEN_IP and LISTEN_PORT
# of the udp_listener.py script that is run by the robot controller.
TARGET_IP = '192.168.37.50' # must be the controller node IP
TARGET_PORT = 13750 # must match the listening port in the controller

def main():
    """
    Main function that creates the socket and manages the sending loop.
    """
    print("--- UDP Dummy Sender ---")
    print(f"Configured to send data to: {TARGET_IP}:{TARGET_PORT}")
    print("Enter a distance value in meters (e.g. 2.5) and press Enter.")
    print("Type 'q' or 'exit' to quit.")
    print("-" * 25)

    # 1. Create the UDP socket
    # AF_INET = IPv4 addresses
    # SOCK_DGRAM = UDP protocol (datagrams)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as e:
        print(f"ERROR: Unable to create socket: {e}")
        sys.exit(1)

    try:
        # 2. Interactive main loop
        while True:
            # Ask for user input
            user_input = input("Distance to send (m): ")

            # Check if the user wants to exit
            if user_input.lower() in ['q', 'exit']:
                print("Exit requested by user.")
                break

            # Try to convert the input to a float
            try:
                distance = float(user_input)
            except ValueError:
                print("Invalid input. Please enter a number (e.g. 1.75).")
                continue  # Skip the rest of the loop and ask for input again

            # 3. Data packing
            # Converts the float number into a sequence of 4 bytes.
            # '<f' means:
            #   < : Little-endian (byte order standard for most architectures)
            #   f : Float (32-bit)
            try:
                packed_data = struct.pack('<f', distance)
            except Exception as e:
                print(f"ERROR: Unable to pack value {distance}: {e}")
                continue

            # 4. Send the UDP packet
            try:
                sock.sendto(packed_data, (TARGET_IP, TARGET_PORT))
                print(f"-> Sent packet with distance: {distance:.2f} m")
            except socket.error as e:
                print(f"ERROR: Send failed: {e}")
                # Add a small pause to avoid flooding the terminal in case of continuous network errors
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt (Ctrl+C). Exiting.")
    finally:
        # 5. Resource cleanup
        print("Closing socket.")
        sock.close()

if __name__ == "__main__":
    main()