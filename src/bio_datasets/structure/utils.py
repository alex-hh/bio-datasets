import io
import logging
import numpy as np
from biotite.structure.io.pdb import PDBFile


logger = logging.getLogger(__name__)

def to_pdb_string(atoms):
    pdb_file = PDBFile()
    if any([len(chain_id) > 1 for chain_id in atoms.chain_id]):
        logger.warning("chain_id is not a single character, truncating")
        atoms.chain_id = np.char.array(atoms.chain_id).astype("U1")
    pdb_file.set_structure(atoms)
    f = io.StringIO()
    pdb_file.write(f)
    return f.getvalue()
