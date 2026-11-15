import json
from unittest.mock import Mock, patch

import pytest

from mem0.vector_stores.cassandra import CassandraDB, OutputData


@pytest.fixture
def mock_session():
    """Create a mock Cassandra session."""
    session = Mock()
    session.execute = Mock(return_value=Mock())
    session.prepare = Mock(return_value=Mock())
    session.set_keyspace = Mock()
    return session


@pytest.fixture
def mock_cluster(mock_session):
    """Create a mock Cassandra cluster."""
    cluster = Mock()
    cluster.connect = Mock(return_value=mock_session)
    cluster.shutdown = Mock()
    return cluster


@pytest.fixture
def cassandra_instance(mock_cluster, mock_session):
    """Create a CassandraDB instance with mocked cluster."""
    with patch('mem0.vector_stores.cassandra.Cluster') as mock_cluster_class:
        mock_cluster_class.return_value = mock_cluster
        
        instance = CassandraDB(
            contact_points=['127.0.0.1'],
            port=9042,
            username='testuser',
            password='testpass',
            keyspace='test_keyspace',
            collection_name='test_collection',
            embedding_model_dims=128,
        )
        instance.session = mock_session
        return instance


def test_cassandra_init(mock_cluster, mock_session):
    """Test CassandraDB initialization."""
    with patch('mem0.vector_stores.cassandra.Cluster') as mock_cluster_class:
        mock_cluster_class.return_value = mock_cluster

        instance = CassandraDB(
            contact_points=['127.0.0.1'],
            port=9042,
            username='testuser',
            password='testpass',
            keyspace='test_keyspace',
            collection_name='test_collection',
            embedding_model_dims=128,
        )

        assert instance.contact_points == ['127.0.0.1']
        assert instance.port == 9042
        assert instance.username == 'testuser'
        assert instance.keyspace == 'test_keyspace'
        assert instance.collection_name == 'test_collection'
        assert instance.embedding_model_dims == 128


def test_create_col(cassandra_instance):
    """Test collection creation."""
    cassandra_instance.create_col(name="new_collection", vector_size=256)

    # Verify that execute was called (table creation)
    assert cassandra_instance.session.execute.called


def test_insert(cassandra_instance):
    """Test vector insertion."""
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    payloads = [{"text": "test1"}, {"text": "test2"}]
    ids = ["id1", "id2"]

    # Mock prepared statement
    mock_prepared = Mock()
    cassandra_instance.session.prepare = Mock(return_value=mock_prepared)

    cassandra_instance.insert(vectors=vectors, payloads=payloads, ids=ids)

    assert cassandra_instance.session.prepare.called
    assert cassandra_instance.session.execute.called


def test_search(cassandra_instance):
    """Test vector search."""
    # Mock the database response
    mock_row1 = Mock()
    mock_row1.id = 'id1'
    mock_row1.vector = [0.1, 0.2, 0.3]
    mock_row1.payload = json.dumps({"text": "test1"})

    mock_row2 = Mock()
    mock_row2.id = 'id2'
    mock_row2.vector = [0.4, 0.5, 0.6]
    mock_row2.payload = json.dumps({"text": "test2"})

    cassandra_instance.session.execute = Mock(return_value=[mock_row1, mock_row2])

    query_vector = [0.2, 0.3, 0.4]
    results = cassandra_instance.search(query="test", vectors=query_vector, top_k=5)

    assert isinstance(results, list)
    assert len(results) <= 5
    assert cassandra_instance.session.execute.called


def test_delete(cassandra_instance):
    """Test vector deletion."""
    mock_prepared = Mock()
    cassandra_instance.session.prepare = Mock(return_value=mock_prepared)

    cassandra_instance.delete(vector_id="test_id")

    assert cassandra_instance.session.prepare.called
    assert cassandra_instance.session.execute.called


def test_update(cassandra_instance):
    """Test vector update."""
    new_vector = [0.7, 0.8, 0.9]
    new_payload = {"text": "updated"}

    mock_prepared = Mock()
    cassandra_instance.session.prepare = Mock(return_value=mock_prepared)

    cassandra_instance.update(vector_id="test_id", vector=new_vector, payload=new_payload)

    assert cassandra_instance.session.prepare.called
    assert cassandra_instance.session.execute.called


def test_get(cassandra_instance):
    """Test retrieving a vector by ID."""
    # Mock the database response
    mock_row = Mock()
    mock_row.id = 'test_id'
    mock_row.vector = [0.1, 0.2, 0.3]
    mock_row.payload = json.dumps({"text": "test"})

    mock_result = Mock()
    mock_result.one = Mock(return_value=mock_row)

    mock_prepared = Mock()
    cassandra_instance.session.prepare = Mock(return_value=mock_prepared)
    cassandra_instance.session.execute = Mock(return_value=mock_result)

    result = cassandra_instance.get(vector_id="test_id")

    assert result is not None
    assert isinstance(result, OutputData)
    assert result.id == "test_id"


def test_list_cols(cassandra_instance):
    """Test listing collections."""
    # Mock the database response
    mock_row1 = Mock()
    mock_row1.table_name = "collection1"

    mock_row2 = Mock()
    mock_row2.table_name = "collection2"

    cassandra_instance.session.execute = Mock(return_value=[mock_row1, mock_row2])

    collections = cassandra_instance.list_cols()

    assert isinstance(collections, list)
    assert len(collections) == 2
    assert "collection1" in collections


def test_delete_col(cassandra_instance):
    """Test collection deletion."""
    cassandra_instance.delete_col()

    assert cassandra_instance.session.execute.called


def test_col_info(cassandra_instance):
    """Test getting collection information."""
    # Mock the database response
    mock_row = Mock()
    mock_row.count = 100

    mock_result = Mock()
    mock_result.one = Mock(return_value=mock_row)

    cassandra_instance.session.execute = Mock(return_value=mock_result)

    info = cassandra_instance.col_info()

    assert isinstance(info, dict)
    assert 'name' in info
    assert 'keyspace' in info


def test_list(cassandra_instance):
    """Test listing vectors."""
    # Mock the database response
    mock_row = Mock()
    mock_row.id = 'id1'
    mock_row.vector = [0.1, 0.2, 0.3]
    mock_row.payload = json.dumps({"text": "test1"})

    cassandra_instance.session.execute = Mock(return_value=[mock_row])

    results = cassandra_instance.list(top_k=10)

    assert isinstance(results, list)
    assert len(results) > 0


def test_reset(cassandra_instance):
    """Test resetting the collection."""
    cassandra_instance.reset()

    assert cassandra_instance.session.execute.called


def test_astra_db_connection(mock_cluster, mock_session):
    """Test connection with DataStax Astra DB secure connect bundle."""
    with patch('mem0.vector_stores.cassandra.Cluster') as mock_cluster_class:
        mock_cluster_class.return_value = mock_cluster

        instance = CassandraDB(
            contact_points=['127.0.0.1'],
            port=9042,
            username='testuser',
            password='testpass',
            keyspace='test_keyspace',
            collection_name='test_collection',
            embedding_model_dims=128,
            secure_connect_bundle='/path/to/bundle.zip'
        )

        assert instance.secure_connect_bundle == '/path/to/bundle.zip'


def test_search_with_filters(cassandra_instance):
    """Test vector search with filters."""
    # Mock the database response
    mock_row1 = Mock()
    mock_row1.id = 'id1'
    mock_row1.vector = [0.1, 0.2, 0.3]
    mock_row1.payload = json.dumps({"text": "test1", "category": "A"})

    mock_row2 = Mock()
    mock_row2.id = 'id2'
    mock_row2.vector = [0.4, 0.5, 0.6]
    mock_row2.payload = json.dumps({"text": "test2", "category": "B"})

    cassandra_instance.session.execute = Mock(return_value=[mock_row1, mock_row2])

    query_vector = [0.2, 0.3, 0.4]
    results = cassandra_instance.search(
        query="test",
        vectors=query_vector,
        top_k=5,
        filters={"category": "A"}
    )

    assert isinstance(results, list)
    # Should only return filtered results
    for result in results:
        assert result.payload.get("category") == "A"


def test_output_data_model():
    """Test OutputData model."""
    data = OutputData(
        id="test_id",
        score=0.95,
        payload={"text": "test"}
    )

    assert data.id == "test_id"
    assert data.score == 0.95
    assert data.payload == {"text": "test"}


def test_insert_without_ids(cassandra_instance):
    """Test vector insertion without providing IDs."""
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    payloads = [{"text": "test1"}, {"text": "test2"}]

    mock_prepared = Mock()
    cassandra_instance.session.prepare = Mock(return_value=mock_prepared)

    cassandra_instance.insert(vectors=vectors, payloads=payloads)

    assert cassandra_instance.session.prepare.called
    assert cassandra_instance.session.execute.called


def test_insert_without_payloads(cassandra_instance):
    """Test vector insertion without providing payloads."""
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ids = ["id1", "id2"]

    mock_prepared = Mock()
    cassandra_instance.session.prepare = Mock(return_value=mock_prepared)

    cassandra_instance.insert(vectors=vectors, ids=ids)

    assert cassandra_instance.session.prepare.called
    assert cassandra_instance.session.execute.called



# --- Metadata filters: operators, logical groups and wildcard ---------------------------------
# Memory._process_metadata_filters() turns user filters into operator dicts ({"age": {"gte": 30}}),
# "$or"/"$not" lists and "*" before calling the vector store. These tests check that
# CassandraDB.search() and CassandraDB.list() honour them instead of silently matching nothing.

FILTER_PAYLOADS = {
    "a": {"user_id": "u1", "age": 20, "city": "Pune", "note": "Likes Tea"},
    "b": {"user_id": "u1", "age": 35, "city": "Delhi", "note": "likes coffee"},
    "c": {"user_id": "u1", "age": 50, "city": "Mumbai"},
    "d": {"user_id": "u2", "age": 40, "city": "Pune", "note": "tea"},
}


def _filter_rows(vector=None):
    rows = []
    for row_id, payload in FILTER_PAYLOADS.items():
        row = Mock()
        row.id = row_id
        row.vector = vector or [1.0, 0.0]
        row.payload = json.dumps(payload)
        rows.append(row)
    return rows


def _matching_ids(instance, method, filters, rows=None):
    instance.session.execute = Mock(return_value=rows if rows is not None else _filter_rows())
    if method == "search":
        results = instance.search(query="q", vectors=[1.0, 0.0], top_k=10, filters=filters)
    else:
        results = instance.list(filters=filters, top_k=10)[0]
    return sorted(r.id for r in results)


FILTER_CASES = [
    pytest.param({"user_id": "u1"}, ["a", "b", "c"], id="plain-equality-still-works"),
    pytest.param({"user_id": "u1", "age": {"gte": 35}}, ["b", "c"], id="gte-includes-the-boundary"),
    pytest.param({"user_id": "u1", "age": {"gt": 35}}, ["c"], id="gt-excludes-the-boundary"),
    pytest.param({"user_id": "u1", "age": {"lte": 35}}, ["a", "b"], id="lte-includes-the-boundary"),
    pytest.param({"user_id": "u1", "age": {"lt": 35}}, ["a"], id="lt-excludes-the-boundary"),
    pytest.param({"user_id": "u1", "age": {"gt": 20, "lt": 50}}, ["b"], id="gt-and-lt-on-one-field"),
    pytest.param({"user_id": "u1", "age": {"eq": 35}}, ["b"], id="eq"),
    pytest.param({"user_id": "u1", "city": {"ne": "Pune"}}, ["b", "c"], id="ne"),
    pytest.param({"user_id": "u1", "city": {"in": ["Pune", "Delhi"]}}, ["a", "b"], id="in"),
    pytest.param({"user_id": "u1", "city": {"nin": ["Pune", "Delhi"]}}, ["c"], id="nin"),
    pytest.param({"user_id": "u1", "city": ["Pune", "Delhi"]}, ["a", "b"], id="list-shorthand-means-in"),
    pytest.param({"user_id": "u1", "note": {"contains": "Tea"}}, ["a"], id="contains-matches-exact-case"),
    pytest.param({"user_id": "u1", "note": {"contains": "tea"}}, [], id="contains-is-case-sensitive"),
    pytest.param({"user_id": "u1", "note": {"icontains": "tea"}}, ["a"], id="icontains-ignores-case"),
    pytest.param({"user_id": "u1", "note": "*"}, ["a", "b"], id="wildcard-means-field-exists"),
    pytest.param({"user_id": "u1", "$or": [{"city": "Pune"}, {"city": "Delhi"}]}, ["a", "b"], id="dollar-or"),
    pytest.param({"user_id": "u1", "OR": [{"city": "Pune"}, {"city": "Delhi"}]}, ["a", "b"], id="or"),
    pytest.param({"user_id": "u1", "$not": [{"city": "Pune"}]}, ["b", "c"], id="dollar-not"),
    pytest.param({"user_id": "u1", "AND": [{"city": "Pune"}, {"age": {"gte": 10}}]}, ["a"], id="and"),
    pytest.param({"user_id": "u1", "city": {"gt": 5}}, [], id="incompatible-types-match-nothing-not-crash"),
]


@pytest.mark.parametrize("method", ["search", "list"])
@pytest.mark.parametrize("filters, expected", FILTER_CASES)
def test_filters_are_applied(cassandra_instance, method, filters, expected):
    assert _matching_ids(cassandra_instance, method, filters) == expected


@pytest.mark.parametrize("method", ["search", "list"])
def test_unsupported_operator_raises_instead_of_matching_nothing(cassandra_instance, method):
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        _matching_ids(cassandra_instance, method, {"user_id": "u1", "age": {"between": [1, 5]}})


@pytest.mark.parametrize("method", ["search", "list"])
def test_mixing_operators_and_unknown_operator_names_raises(cassandra_instance, method):
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        _matching_ids(cassandra_instance, method, {"user_id": "u1", "$or": [{"age": {"bogus": 1}}]})


def test_search_scores_a_zero_vector_row_as_zero_not_nan(cassandra_instance):
    row = Mock()
    row.id = "z"
    row.vector = [0.0, 0.0]
    row.payload = json.dumps({"user_id": "u1"})
    cassandra_instance.session.execute = Mock(return_value=[row])
    results = cassandra_instance.search(query="q", vectors=[1.0, 0.0], top_k=5, filters={"user_id": "u1"})
    assert [r.score for r in results] == [0.0]
