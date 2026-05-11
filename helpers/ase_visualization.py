import py3Dmol
from ase.io import write
from io import StringIO


def ase_to_xyz_string(atoms):
    """
    Convert ASE Atoms object to xyz string.
    """
    sio = StringIO()
    write(sio, atoms, format="xyz")
    return sio.getvalue()


def show_structure(atoms, style="stick", width=500, height=400):
    """
    Visualize single ASE structure with py3Dmol.
    """

    xyz = ase_to_xyz_string(atoms)

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz, "xyz")
    view.setStyle({style: {}})
    view.zoomTo()

    return view.show()


def show_neb(images, width=900, height=400):
    """
    Visualize all NEB images.
    """

    view = py3Dmol.view(width=width, height=height)

    for atoms in images:
        xyz = ase_to_xyz_string(atoms)
        view.addModel(xyz, "xyz")

    view.setStyle({"stick": {}})
    view.zoomTo()

    return view.show()

