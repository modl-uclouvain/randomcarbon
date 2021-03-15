# example script to execute deduplication of results in a mongodb collection
# this can run in parallel using multiprocessing. List of structures will be
# split based on generic properties (number of sites, range of energies per atoms)
# and subsequently matched with a StructureMatcher.
# The elements of the collection that are duplicated of another will be marked
# with a "duplicated" properties, having the structure_id of the equivalent structure.

from randomcarbon.output.deduplicate import deduplicate_data_paral

conn_data = dict(host='host', port=27017, username='username', password='password',
                 database='database_name', collection_name="collection_name")

deduplicate_data_paral(n_procs=10, connection_data=conn_data)
