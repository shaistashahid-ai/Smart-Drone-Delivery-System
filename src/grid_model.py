"""
# Grid Model
Shared 10x10 city grid. Each cell holds zone, density, flags, and state.
"""

from dataclasses import dataclass,field
from typing import List,Dict,Tuple
import random

# --- Cell dataclass ---
@dataclass
class Cell:
    """Single grid cell with zone info and operational flags."""
    row:int=0
    col:int=0
    zone:str="OpenField"           # Residential,Commercial,Industrial,Hospital,School,OpenField
    density:int=0                  # population density
    isHub:bool=False               # drone hub
    isCharging:bool=False          # charging pad
    isMedPickup:bool=False         # medical supply pickup
    noFly:bool=False               # blocked for routing
    demand:float=0.0               # estimated delivery demand

# --- Drone dataclass ---
@dataclass
class Drone:
    """Drone unit with type, payload, range, and state."""
    dId:str=""                     # e.g. D1, D2
    dType:str="Light"              # Light or Heavy
    cost:int=1000
    payload:float=2.0              # kg
    maxRange:int=12                # cells
    pos:Tuple[int,int]=(0,0)      # current position
    homeHub:Tuple[int,int]=(0,0)  # assigned hub
    battery:float=100.0
    status:str="idle"              # idle,enroute,returning,grounded

# --- Delivery dataclass ---
@dataclass
class Delivery:
    """Single delivery task."""
    delId:str=""                   # e.g. DEL1
    pickup:Tuple[int,int]=(0,0)
    dropoff:Tuple[int,int]=(0,0)
    weight:float=1.0              # kg
    assignedDrone:str=""
    status:str="pending"          # pending,inprogress,completed,delayed,failed
    route:List[Tuple[int,int]]=field(default_factory=list)
    routeCost:float=0.0

# --- Grid builder ---
def buildGrid()->List[List[Cell]]:
    """Build a predefined 10x10 city grid with realistic zone layout."""
    # zone layout map
    zMap=[
        ["Commercial","Commercial","Residential","Residential","OpenField","OpenField","Residential","Residential","Commercial","Commercial"],
        ["Commercial","Hospital","Residential","Residential","School","OpenField","Residential","Residential","Industrial","Commercial"],
        ["Residential","Residential","Residential","OpenField","OpenField","OpenField","Residential","Residential","Industrial","Industrial"],
        ["Residential","Residential","OpenField","Commercial","Commercial","Commercial","OpenField","Residential","Residential","Residential"],
        ["OpenField","School","OpenField","Commercial","Hospital","Commercial","OpenField","OpenField","Residential","Residential"],
        ["Residential","Residential","Residential","OpenField","OpenField","OpenField","Commercial","Commercial","OpenField","OpenField"],
        ["Residential","Residential","Residential","Residential","OpenField","Residential","Residential","Commercial","Industrial","OpenField"],
        ["OpenField","Residential","Residential","Residential","School","Residential","Residential","OpenField","OpenField","OpenField"],
        ["Industrial","OpenField","Residential","Residential","Residential","Residential","Residential","Residential","OpenField","Commercial"],
        ["Industrial","Industrial","OpenField","OpenField","Residential","Residential","OpenField","Commercial","Commercial","Commercial"],
    ]
    # density map (population per cell)
    dMap=[
        [3000,3500,5000,4800,200,200,4500,4700,3200,3400],
        [3200,1000,5200,5100,800,200,4900,5000,1500,3100],
        [5500,5300,5100,300,200,200,4800,4600,1200,1000],
        [4900,4700,300,3800,4000,3900,300,4500,4300,4100],
        [200,900,200,3600,1200,3700,200,200,4800,4600],
        [5000,4800,4600,300,200,200,3500,3300,200,200],
        [5200,5000,4900,4700,200,4500,4300,3100,1100,200],
        [200,4600,4400,4200,700,4000,3800,200,200,200],
        [1300,200,4100,3900,3700,3500,3300,3100,200,2800],
        [1100,1000,200,200,3200,3000,200,2900,3100,3300],
    ]

    grid=[]
    for r in range(10):
        row=[]
        for c in range(10):
            cell=Cell(
                row=r,col=c,
                zone=zMap[r][c],
                density=dMap[r][c],
                demand=round(dMap[r][c]*random.uniform(0.0005,0.002),1)
            )
            row.append(cell)
        grid.append(row)

    # --- Place hubs ---
    hubs=[(1,1),(4,4),(7,5),(9,8),(0,7),(3,9),(6,1),(8,3)]
    for r,c in hubs:
        grid[r][c].isHub=True

    # --- Place charging pads ---
    chPads=[(1,2),(0,1),(4,3),(3,4),(7,4),(7,6),(9,7),(9,9),(0,6),(3,8),(6,0),(8,2)]
    for r,c in chPads:
        grid[r][c].isCharging=True

    # --- Place medical pickups ---
    medPts=[(1,2),(4,3)]
    for r,c in medPts:
        grid[r][c].isMedPickup=True

    # --- Initial no-fly cells ---
    nfCells=[(2,8),(8,0)]
    for r,c in nfCells:
        grid[r][c].noFly=True

    return grid

def getNeighbors(r:int,c:int,sz:int=10)->List[Tuple[int,int]]:
    """Return valid 4-directional neighbors."""
    nbrs=[]
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr,nc=r+dr,c+dc
        if 0<=nr<sz and 0<=nc<sz:
            nbrs.append((nr,nc))
    return nbrs

def manhattan(a:Tuple[int,int],b:Tuple[int,int])->int:
    """Manhattan distance between two cells."""
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def printGrid(grid:List[List[Cell]]):
    """Print zone abbreviations for quick view."""
    abbrv={"Residential":"RES","Commercial":"COM","Industrial":"IND",
           "Hospital":"HOS","School":"SCH","OpenField":"OPN"}
    for row in grid:
        print(" ".join(f"{abbrv.get(c.zone,'???'):>3}" for c in row))
