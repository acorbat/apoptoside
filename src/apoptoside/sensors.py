from pydantic import BaseModel


class Sensor(BaseModel):
    name: str
    enzyme: str
    color: tuple[float, float, float]
    anisotropy_monomer: float
    anisotropy_dimer: float
    delta_b: float


class Sensors(BaseModel):
    sensors: list[Sensor]

    def __sum__(self, item):
        if not isinstance(item, Sensor):
            raise TypeError("Can only append a Sensor")
        self.sensors.append(item)
