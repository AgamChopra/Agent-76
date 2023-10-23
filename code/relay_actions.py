"""
Created on October 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
"""
import serial.tools.list_ports


class Relay():
    def __init__(self, channel=9600):
        self.channel = channel
        self.ports = serial.tools.list_ports.comports()
        self.serialInst = serial.Serial()
        self.portsList = []
        for onePort in self.ports:
            self.portsList.append(str(onePort))

        self.portVar = None

    def lookup_ports(self):
        for onePort in self.portsList:
            print(onePort)

    def activate_port(self, port_number):
        for p in self.portsList:
            if p.startswith("COM" + str(port_number)):
                self.portVar = "COM" + str(port_number)

        self.serialInst.baudrate = self.channel
        self.serialInst.port = self.portVar
        self.serialInst.open()

    def free_port(self):
        self.serialInst.close()

    def send_action(self, action='0,0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'):
        self.serialInst.write(action.encode('utf-8'))


def test():
    relay = Relay()
    relay.lookup_ports()
    relay.activate_port(input('Port #: '))

    while (True):
        command = input("Command: ")
        relay.send_action(command)

        if command == 'p':
            relay.free_port()
            break


if __name__ == '__main__':
    test()
