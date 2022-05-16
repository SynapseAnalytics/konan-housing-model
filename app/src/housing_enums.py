from enum import Enum


class DetailedRatingTypes(int, Enum):
    VeryPoor = 1
    Poor = 2
    Fair = 3
    BelowAverage = 4
    Average = 5
    AboveAverage = 6
    Good = 7
    VeryGood = 8
    Excellant = 9
    VeryExcellant = 10


class RatingTypes(str, Enum):
    Excellent = 'Ex'
    Good = 'Gd'
    Typical = 'TA'
    Fair = 'Fa'
    Poor = 'Po'


class FoundationTypes(str, Enum):
    Brick = 'BrkTil'
    Cinder = 'CBlock'
    Concret = 'PConc'
    Slab = 'Slab'
    Stone = 'Stone'
    Wood = 'Wood'


class HeatingTypes(str, Enum):
    Floor = 'Floor'
    GasAir = 'GasA'
    GasWater = 'GasWater'
    GravityFurnace = 'Grav'
    HotWater = 'OthW'
    WallFurnace = 'Wall'


class FunctionalTypes(str, Enum):
    Typical = 'Typ'
    VeryMinorDeductions = 'Min1'
    MinorDeductions = 'Min2'
    ModerateDeductions = 'Mod'
    MajorDeductions = 'Maj1'
    VeryMajorDeductions = 'Maj2'
    SeverelyDamaged = 'Sev'
    SalvageOnly = 'Sal'


class DrivewayTypes(str, Enum):
    Paved = 'Y'
    PartiallyPaved = 'P'
    Gravel = 'N'


class FenceQualityTypes(str, Enum):
    GoodPrivacy = 'GdPrv'
    MinimumPrivacy = 'MnPrv'
    GoodWood = 'GdWo'
    MinimumWood = 'MnWw'


class MiscellaneousFeaturesTypes(str, Enum):
    Elevator = 'Elev'
    SecondGarage = 'Gar2'
    Shed = 'Shed'
    TennisCourt = 'TenC'
    Other = 'Othr'
