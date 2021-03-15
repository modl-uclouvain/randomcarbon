# example script to add tags to existing results in an existing Store.
# Here only a simple tagger is used. Typical usage include adding tags that may
# be computationally intensive and could be interesting only for the more
# interesting candidates.
from randomcarbon.output.results import tag_results
from randomcarbon.output.store import MongoStore
from randomcarbon.output.taggers import SymmetryTag

store = MongoStore(host='host', port=27017, username='username', password='password',
                   database='database_name', collection_name="collection_name")
taggers = [SymmetryTag()]
tag_results(store=store, criteria={"energy_per_atoms": {"$lte": -6.5}}, taggers=taggers)
