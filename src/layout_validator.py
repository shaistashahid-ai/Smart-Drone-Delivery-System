"""
# Layout Validator (CSP)
Checks grid against 4 constraint rules. Reports violations with cell coords.
"""

from typing import List,Dict,Tuple
from src.grid_model import Cell,getNeighbors,manhattan

def chkIndustrialSafety(grid:List[List[Cell]])->List[str]:
    """R1: Industrial cells cannot be adjacent to Schools or Hospitals."""
    errs=[]
    banned={"School","Hospital"}
    for row in grid:
        for cell in row:
            if cell.zone!="Industrial":
                continue
            for nr,nc in getNeighbors(cell.row,cell.col):
                nbr=grid[nr][nc]
                if nbr.zone in banned:
                    errs.append(f"R1 FAIL: Industrial({cell.row},{cell.col}) adjacent to {nbr.zone}({nr},{nc})")
    return errs

def chkResidentialCoverage(grid:List[List[Cell]])->List[str]:
    """R2: Every Residential cell must be within 3 Manhattan of a hub."""
    errs=[]
    # collect hubs
    hubs=[(c.row,c.col) for row in grid for c in row if c.isHub]
    for row in grid:
        for cell in row:
            if cell.zone!="Residential":
                continue
            minDist=min((manhattan((cell.row,cell.col),h) for h in hubs),default=999)
            if minDist>3:
                errs.append(f"R2 FAIL: Residential({cell.row},{cell.col}) is {minDist} cells from nearest hub. Suggest hub near ({cell.row},{cell.col})")
    return errs

def chkHubCharging(grid:List[List[Cell]])->List[str]:
    """R3: Every hub must have a charging pad within 2 cells."""
    errs=[]
    chPads=[(c.row,c.col) for row in grid for c in row if c.isCharging]
    hubs=[(c.row,c.col) for row in grid for c in row if c.isHub]
    for h in hubs:
        minDist=min((manhattan(h,cp) for cp in chPads),default=999)
        if minDist>2:
            errs.append(f"R3 FAIL: Hub({h[0]},{h[1]}) has no charging pad within 2 cells")
    return errs

def chkMedicalAccess(grid:List[List[Cell]])->List[str]:
    """R4: At least one Hospital must have a medical pickup within 1 cell."""
    errs=[]
    hosps=[(c.row,c.col) for row in grid for c in row if c.zone=="Hospital"]
    medPts=[(c.row,c.col) for row in grid for c in row if c.isMedPickup]
    found=False
    for h in hosps:
        for mp in medPts:
            if manhattan(h,mp)<=1:
                found=True
                break
        if found:
            break
    if not found:
        errs.append("R4 FAIL: No Hospital has a Medical Pickup within 1 cell")
    return errs

def validateLayout(grid:List[List[Cell]])->Dict:
    """Run all 4 CSP rules. Return results dict."""
    results={"passed":[],"failed":[],"errors":[]}
    rules=[
        ("R1",chkIndustrialSafety),
        ("R2",chkResidentialCoverage),
        ("R3",chkHubCharging),
        ("R4",chkMedicalAccess),
    ]
    for rId,fn in rules:
        errs=fn(grid)
        if errs:
            results["failed"].append(rId)
            results["errors"].extend(errs)
        else:
            results["passed"].append(rId)

    # print report
    print("="*60)
    print("LAYOUT VALIDATION REPORT")
    print("="*60)
    valid=len(results["failed"])==0
    print(f"Layout validity = {valid}")
    print(f"Passed rules: {', '.join(results['passed']) if results['passed'] else 'None'}")
    print(f"Failed rules: {', '.join(results['failed']) if results['failed'] else 'None'}")
    if results["errors"]:
        print("\nViolations:")
        for e in results["errors"]:
            print(f"  - {e}")
    print("="*60)
    return results
