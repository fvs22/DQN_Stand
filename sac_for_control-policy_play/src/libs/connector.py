import socket
import struct
from typing import Tuple
from numpy import random
import datetime


class AbstractConnector:
    def __init__(self, address: Tuple[str, int]):
        print('Start')
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(address)
        self.socket.listen(1)
        self.connection, self.client_address = self.socket.accept()
        print('End binding')

    def receive(self, y_target):
        pass

    def step(self, action, flag, y_target):
        pass

    def close(self):
        pass

    def reset_simulation(self, simulation_transfer: int):
        pass


class RealConnector(AbstractConnector):
    def __init__(self, address: Tuple[str, int]):
        super().__init__(address)

    def receive(self, y_target):
        try:
            data = self.connection.recv(24)
        except Exception:
            self.connection.close()
        i, l, v, metric, done, object_velocity = struct.unpack('ffffff', data)
        l = abs(l / 100)
        object_velocity = object_velocity / 100
        state = [i, l, v, object_velocity, y_target]
        # state_to_output.append([*state, datetime.datetime.now()])
        # print(state)
        return state, metric, int(done)

    def step(self, action, flag, y_target):
        # print('send ', float(action))
        try:
            # print(action)
            self.connection.send(struct.pack('fff', float(action), float(flag), float(y_target)))
        except Exception:
            self.connection.close()

    def close(self):
        self.connection.close()


class Connector(AbstractConnector):
    def __init__(self, address: Tuple[str, int]):
        super().__init__(address)

    def receive(self):
        data = self.connection.recv(56)
        # print(data)

        i, l, v, metric, done, y_target, object_velocity = struct.unpack('ddddddd', data)
        state = [i, l, v, object_velocity]
        # print(state)
        return state, metric, y_target, int(done)

    def reset_simulation(self, simulation_transfer: int):
        for step in range(simulation_transfer):
            self.step(0.1)
            _, _, _, _ = self.receive()

    def step(self, action):
        # print(float(action))
        self.connection.send(struct.pack('d', float(action)))

    def close(self):
        self.connection.close()
