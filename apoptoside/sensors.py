from typing import List, Tuple
from datastruct import DataStruct


class Sensor(DataStruct):
    name: str
    enzyme: str
    color: Tuple[float]
    anisotropy_monomer: float
    anisotropy_dimer: float
    delta_b: float


class Sensors(DataStruct):
    sensors: List[Sensor]

    def __sum__(self, item):
        if not isinstance(item, Sensor):
            raise TypeError('Can only append a Sensor')
        self.sensors.append(item)
