from __future__ import annotations

import pyarrow as pa
import pytest

import tiledbsoma as soma

from tests._util import maybe_raises

# ================================================================
# TODO:

# ----------------------------------------------------------------
# * snda create w/ shape & maxshape
#   o UT shape <, ==, > maxshape
# * dnda ditto
#   o UT shape <, ==, > maxshape
# * sdf ditto
#   o UT shape <, ==, > maxshape
#   o UT partials w/ extra dims

# ----------------------------------------------------------------
# * all 3:
#   o UT OOB writes
#   o UT OOB reads

# ----------------------------------------------------------------
# * used_shape accessor:
#   o as-is
#   o deprecation notice ...
# * shape accessor:
#   o ret new if avail
#   o else ret old w/ deprecation notice ... ?
# * ned accessor
#   o as is
# * maxshape accessor
#   o new
#   ! spell it maxshape

# ----------------------------------------------------------------
# * resize mutator
#   o NotImplementedError for old arrays
#   o ValueError if shrinking CD
#   o ValueError if bigger than domain

# ----------------------------------------------------------------
# * tiledbsoma_upgrade_shape for snda/dnda
#   o array.schema.version to see if needed
#   o use core storage-version-update logic ...
#   o fail if outside domain
# * tiledbsoma_upgrade_shape for sdf
#   o arg name is domain not shape

# ----------------------------------------------------------------
# * tiledbsoma.io.resize ...
#   o per array
#   o do-it-all w/ new nobs/nvar -- ?
# ================================================================


@pytest.mark.parametrize(
    "shape_maxshape_exc",
    [
        [(100, 200), None, None],
        [(100, 200), (None, None), None],
        [(100, 200), (100, 200), None],
        [(100, 200), (1000, 200), None],
        [(100, 200), (100, 2000), None],
        [(100, 200), (10, 200), soma.SOMAError],
        [(100, 200), (10, 200), soma.SOMAError],
        [(100, 200), (100,), ValueError],
        [(100, 200), (100, 200, 300), ValueError],
    ],
)
def test_sparse_nd_array_create(
    tmp_path,
    shape_maxshape_exc,
):
    shape, maxshape, exc = shape_maxshape_exc
    uri = tmp_path.as_posix()
    element_type = pa.float32()

    # Create the array
    with maybe_raises(exc):
        snda = soma.SparseNDArray.create(
            uri, type=element_type, shape=shape, maxshape=maxshape
        )
    if exc is not None:
        return

    assert soma.SparseNDArray.exists(uri)

    # Test the various accessors
    with soma.SparseNDArray.open(uri) as snda:
        assert snda.shape == shape

        # TODO: need a saved-off array in UT-data land

        # If maxshape is None, or None in any slot, we expect it to be set to a
        # big signed int32. (There are details on the exact value of that
        # number, involving R compatibility, and leaving room for a single tile
        # capacity, etc ...  we could check for some magic value but it suffices
        # to check that it's over 2 billion.)
        if maxshape is None:
            for e in snda.maxshape:
                assert e > 2_000_000_000
        else:
            for i in range(len(shape)):
                if maxshape[i] is None:
                    assert snda.maxshape[i] > 2_000_000_000
                else:
                    assert snda.maxshape[i] == maxshape[i]

        # TODO: used_shape
        #   o as-is
        #   o deprecation notice ...

        # No data have been written for this test case
        assert snda.non_empty_domain() == ((0, 0), (0, 0))

    # Write some data
    with soma.SparseNDArray.open(uri, "w") as snda:
        table = pa.Table.from_pydict(
            {
                "soma_dim_0": [0, 1],
                "soma_dim_1": [2, 3],
                "soma_data": [4, 5],
            }
        )
        snda.write(table)

    # Test the various accessors
    with soma.SparseNDArray.open(uri) as snda:
        assert snda.shape == shape
        if maxshape is None:
            for e in snda.maxshape:
                assert e > 2_000_000_000
        else:
            for i in range(len(shape)):
                if maxshape[i] is None:
                    assert snda.maxshape[i] > 2_000_000_000
                else:
                    assert snda.maxshape[i] == maxshape[i]
        assert snda.non_empty_domain() == ((0, 1), (2, 3))

    # Test reads out of bounds
    with soma.SparseNDArray.open(uri) as snda:
        with pytest.raises(soma.SOMAError):
            coords = ((shape[0] + 10,), (shape[1] + 20,))
            snda.read(coords)

    # Test writes out of bounds
    with soma.SparseNDArray.open(uri, "w") as snda:
        with pytest.raises(soma.SOMAError):
            table = pa.Table.from_pydict(
                {
                    "soma_dim_0": [shape[0] + 10],
                    "soma_dim_1": [shape[1] + 20],
                    "soma_data": [30],
                }
            )
            snda.write(table)


# ----------------------------------------------------------------
# XXX DNDA all

# ----------------------------------------------------------------
# XXX SDF all
# XXX partials w/ extra dims

# ----------------------------------------------------------------
# * resize mutator
#   o NotImplementedError for old arrays
#   o ValueError if shrinking CD
#   o ValueError if bigger than domain

# ----------------------------------------------------------------
# * tiledbsoma_upgrade_shape for snda/dnda
#   o array.schema.version to see if needed
#   o use core storage-version-update logic ...
#   o fail if outside domain
# * tiledbsoma_upgrade_shape for sdf
#   o arg name is domain not shape

# ----------------------------------------------------------------
# * tiledbsoma.io.resize ...
#   o per array
#   o do-it-all w/ new nobs/nvar -- ?
