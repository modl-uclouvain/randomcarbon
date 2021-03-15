from abc import ABCMeta, abstractmethod
from typing import Union, List, Dict, Optional, Iterator
from pymongo import MongoClient, InsertOne, ReplaceOne, uri_parser
from pymongo.errors import OperationFailure, DocumentTooLarge, ConfigurationError
import json
import os
from monty.io import zopen
from monty.os import makedirs_p
from monty.serialization import loadfn, dumpfn
from monty.json import jsanitize, MSONable


class Store(MSONable, metaclass=ABCMeta):
    """
    Base class defining an object to store results in different kind of targets.
    """

    def __init__(self, key: str = "structure_id"):
        self.key = key

    @abstractmethod
    def connect(self):
        """
        Method that will perform the procedure required to access the store.
        """
        pass

    def insert(self, docs: Union[List[Dict], Dict]):
        """
        Update documents into the Store

        Args:
            docs: the document or list of documents to update
        """

    @abstractmethod
    def close(self):
        """
        Closes any connections
        """

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class MongoStore(Store):
    """
    A Store that connects to a Mongo collection.
    Modelled on maggma MongoStore.
    """

    def __init__(
        self,
        database: str,
        collection_name: str,
        host: str = "localhost",
        port: int = 27017,
        username: str = "",
        password: str = "",
        **kwargs,
    ):
        """
        Args:
            database: The database name
            collection_name: The collection name
            host: Hostname for the database
            port: TCP port to connect to
            username: Username for the collection
            password: Password to connect with
        """
        self.database = database
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._collection = None  # type: Any
        super().__init__(**kwargs)

    def connect(self):
        """
        Connect to the source data
        """
        if not self._collection:
            conn = MongoClient(self.host, self.port)
            db = conn[self.database]
            if self.username != "":
                db.authenticate(self.username, self.password)
            self._collection = db[self.collection_name]

    @property
    def collection(self):
        """
        Expose the internal collection object.
        None if the Store is not connected.
        """
        return self._collection

    @classmethod
    def from_db_file(cls, filename: str):
        """
        Convenience method to construct MongoStore from db_file
        from old QueryEngine format
        """
        kwargs = loadfn(filename)
        if "collection" in kwargs:
            kwargs["collection_name"] = kwargs.pop("collection")
        # Get rid of aliases from traditional query engine db docs
        kwargs.pop("aliases", None)
        return cls(**kwargs)

    def insert(self, docs: Union[List[Dict], Dict]):
        """
        Update documents into the Store

        Args:
            docs: the document or list of documents to update
        """

        requests = []

        if not isinstance(docs, list):
            docs = [docs]

        for d in docs:

            d = jsanitize(d, allow_bson=True)
            requests.append(InsertOne(d))

        if len(requests) > 0:
            self._collection.bulk_write(requests, ordered=False)

    def close(self):
        """ Close up all collections """
        self._collection.database.client.close()

    def query(
        self,
        criteria: Optional[Dict] = None,
        properties: Union[Dict, List, None] = None,
        sort: Optional[Dict[str, int]] = None,
        skip: int = 0,
        limit: int = 0,
    ) -> Iterator[Dict]:
        """
        Queries the Store for a set of documents
        Args:
            criteria: PyMongo filter for documents to search in
            properties: properties to return in grouped documents
            sort: Dictionary of sort order for fields. Keys are field names and
                values are 1 for ascending or -1 for descending.
            skip: number documents to skip
            limit: limit on total number of documents returned
        """
        if isinstance(properties, list):
            properties = {p: 1 for p in properties}

        sort_list = (
            list(sort.items()) if sort else None
        )

        for d in self._collection.find(
            filter=criteria,
            projection=properties,
            skip=skip,
            limit=limit,
            sort=sort_list,
        ):
            yield d

    def update(self, docs: Union[List[Dict], Dict], key: Union[List, str, None] = None):
        """
        Update documents into the Store
        Args:
            docs: the document or list of documents to update
            key: field name(s) to determine uniqueness for a
                 document, can be a list of multiple fields,
                 a single field, or None if the Store's key
                 field is to be used
        """

        requests = []

        if not isinstance(docs, list):
            docs = [docs]

        for d in docs:

            d = jsanitize(d, allow_bson=True)

            key = key or self.key
            if isinstance(key, list):
                search_doc = {k: d[k] for k in key}
            else:
                search_doc = {key: d[key]}

            requests.append(ReplaceOne(search_doc, d, upsert=True))

        if len(requests) > 0:
            self._collection.bulk_write(requests, ordered=False)

    def remove_docs(self, criteria: Dict):
        """
        Remove docs matching the query dictionary
        Args:
            criteria: query dictionary to match
        """
        self._collection.delete_many(filter=criteria)


class MultiJsonStore(Store):
    """
    A store that writes each document in a json file in the specified folder.
    The name of the folder will be the self.key value. Thus the value should be
    unique.
    """

    def __init__(self, folder: str, **kwargs):
        self.folder = os.path.abspath(folder)
        super().__init__(**kwargs)

    def connect(self):
        """
        Creates the target folder if not already existing.
        """
        makedirs_p(self.folder)

    def close(self):
        pass

    def insert(self, docs: Union[List[Dict], Dict]):
        """
        Writes one file for each document to the folder.

        Args:
            docs: the document or list of documents to update
        """

        if not isinstance(docs, list):
            docs = [docs]

        for d in docs:
            d = jsanitize(d, allow_bson=False)
            key = d[self.key]
            dumpfn(d, os.path.join(self.folder, f"{key}.json.gz"))


class JsonFileStore(Store):
    """
    Store based on a single json file.
    The data is taken from the file when connect() is called and
    written down when close() is called. In the meanwhile the data
    will just stay in memory. Does not support multiprocessing.
    """

    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        self.data = None
        super().__init__(**kwargs)

    def connect(self):
        if self.data is not None:
            raise RuntimeError("File already opened. Close the store before to dump data.")

        if os.path.isfile(self.filepath):
            with zopen(self.filepath) as f:
                self.data = json.load(f)

        else:
            self.data = {}

    def close(self):

        if self.data is not None:
            with zopen(self.filepath, "wt") as f:
                json.dump(self.data, f)

    def insert(self, docs: Union[List[Dict], Dict]):
        """
        Writes one file for each document to the folder.

        Args:
            docs: the document or list of documents to update
        """

        if self.data is None:
            raise RuntimeError("The store should be connected before inserting.")

        if not isinstance(docs, list):
            docs = [docs]

        for d in docs:
            d = jsanitize(d, allow_bson=False)
            key = d.pop(self.key)
            self.data[key] = d
