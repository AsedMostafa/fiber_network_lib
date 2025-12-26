import sys
from pathlib import Path

def read_bounds(lines):
    for line in lines:
        if "xlo xhi" in line:
            p = line.split()
            return float(p[0]), float(p[1])
    raise RuntimeError("xlo xhi not found")

def find_section(lines, name):
    for i, line in enumerate(lines):
        if line.strip().startswith(name):
            return i
    return -1

def extract_section(lines, start_idx, count):
    return lines[start_idx+2 : start_idx+2+count]

def shift_atoms(atom_lines, dx, id_shift):
    out = []
    for line in atom_lines:
        parts = line.split()
        aid = int(parts[0]) + id_shift
        x = float(parts[3]) + dx
        parts[0] = str(aid)
        parts[3] = f"{x:.6f}"
        out.append(" ".join(parts))
    return out

def shift_vels(vel_lines, id_shift):
    out = []
    for line in vel_lines:
        parts = line.split()
        parts[0] = str(int(parts[0]) + id_shift)
        out.append(" ".join(parts))
    return out

def shift_bonds(bond_lines, bond_id_shift, atom_id_shift):
    out = []
    for line in bond_lines:
        p = line.split()
        bid  = int(p[0]) + bond_id_shift
        a1   = int(p[2]) + atom_id_shift
        a2   = int(p[3]) + atom_id_shift
        p[0] = str(bid)
        p[2] = str(a1)
        p[3] = str(a2)
        out.append(" ".join(p))
    return out

# -------------------------------------------------------------

def concat_data_files(files, order, outfile):

    # reorder files by order list (1-based)
    ordered_files = [files[i-1] for i in order]

    # Storage for combined system
    combined_atoms = []
    combined_vels  = []
    combined_bonds = []

    # We will copy Masses section from the first file
    final_header = None
    masses = None

    total_atoms = 0
    total_bonds = 0
    current_xmax = None

    for idx, fpath in enumerate(ordered_files):
        lines = Path(fpath).read_text().splitlines()

        # Header info
        # --------------------------------------
        # counts
        natoms = nbonds = None
        for line in lines[:20]:
            s = line.strip().split()
            if len(s) >= 2:
                if s[1] == "atoms": natoms = int(s[0])
                if s[1] == "bonds": nbonds = int(s[0])
        if natoms is None or nbonds is None:
            raise RuntimeError("Could not read atom/bond counts")

        # box bounds
        xlo, xhi = read_bounds(lines)
        Lx = xhi - xlo

        # indices
        i_mass = find_section(lines, "Masses")
        i_atoms = find_section(lines, "Atoms")
        i_vel   = find_section(lines, "Velocities")
        i_bonds = find_section(lines, "Bonds")

        # Save Masses only once
        if idx == 0:
            final_header = lines[:i_mass]
            masses = []
            j = i_mass + 2
            while j < len(lines) and lines[j].strip():
                masses.append(lines[j])
                j += 1

        # extract raw sections
        atoms_raw = extract_section(lines, i_atoms, natoms)
        vels_raw  = extract_section(lines, i_vel,   natoms)
        bonds_raw = extract_section(lines, i_bonds, nbonds)

        # compute shifts
        atom_id_shift = total_atoms
        bond_id_shift = total_bonds
        dx = current_xmax if current_xmax is not None else 0

        # shift blocks
        atoms_shifted = shift_atoms(atoms_raw, dx, atom_id_shift)
        vels_shifted  = shift_vels(vels_raw,    atom_id_shift)
        bonds_shifted = shift_bonds(bonds_raw,  bond_id_shift, atom_id_shift)

        # accumulate
        combined_atoms.extend(atoms_shifted)
        combined_vels.extend(vels_shifted)
        combined_bonds.extend(bonds_shifted)

        # update counters
        total_atoms += natoms
        total_bonds += nbonds
        current_xmax = dx + Lx  # where the next system begins

    # Build final header with updated atom/bond counts and expanded box
    for i,line in enumerate(final_header):
        s = line.strip()
        if s.endswith("atoms"):
            final_header[i] = f"{total_atoms} atoms"
        elif s.endswith("bonds"):
            final_header[i] = f"{total_bonds} bonds"
        elif "xlo xhi" in s:
            # extend xhi
            xlo0 = float(s.split()[0])
            final_header[i] = f"{xlo0} {xlo0 + current_xmax} xlo xhi"

    # write output
    out_lines = []
    out_lines.extend(final_header)
    out_lines.append("")
    out_lines.append("Masses")
    out_lines.append("")
    out_lines.extend(masses)
    out_lines.append("")

    out_lines.append("Atoms # angle")
    out_lines.append("")
    out_lines.extend(combined_atoms)
    out_lines.append("")

    out_lines.append("Velocities")
    out_lines.append("")
    out_lines.extend(combined_vels)
    out_lines.append("")

    out_lines.append("Bonds")
    out_lines.append("")
    out_lines.extend(combined_bonds)
    out_lines.append("")

    Path(outfile).write_text("\n".join(out_lines))
    print(f"Written: {outfile}")

# -------------------------------------------------------------
# Example usage:
# python concat_lammps_multi.py data1 data2 data3 "1,3,2" output.data
# -------------------------------------------------------------
if __name__ == "__main__":
    files = sys.argv[1:-2]
    order = list(map(int, sys.argv[-2].split(",")))
    outfile = sys.argv[-1]
    concat_data_files(files, order, outfile)
