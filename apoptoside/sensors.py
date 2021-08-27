from typing import List, Tuple
from pydantic import BaseModel


class Sensor(BaseModel):
    name: str
    enzyme: str
    color: Tuple[float, float, float]
    anisotropy_monomer: float
    anisotropy_dimer: float
    delta_b: float


class Sensors(BaseModel):
    sensors: List[Sensor]

    def __sum__(self, item):
        if not isinstance(item, Sensor):
            raise TypeError('Can only append a Sensor')
        self.sensors.append(item)
