from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dmcp.initial import rand_initial
from dmcp.initial import rand_initial_proj
from dmcp.fix import fix
from dmcp.utils import is_atom_multiconvex
from dmcp.find_set import find_minimal_sets
from dmcp.bcd import is_dmcp
from dmcp.bcd import bcd
from dmcp.bcd import linearize
from dmcp.bcd import add_slack
from dmcp.bcd import proximal_op

__author__ = 'Xinyue'
__version___ = "1.0.0"