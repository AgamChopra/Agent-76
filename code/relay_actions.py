"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import serial.tools.list_ports


ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portsList = []

for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))

val = input("Serial Port: COM")

for p in portsList:
    if p.startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portVar)

serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()

while (True):
    command = input("Command: ")
    serialInst.write(command.encode('utf-8'))

    if command == 'p':
        serialInst.close()
        break