from app.services.rag_service import extract_price_info, extract_food_query

def test_extract_price_info_under_50k():
    min_p, max_p, query = extract_price_info("món dưới 50k")
    assert max_p == 50000.0
    assert "món" in query

def test_extract_price_info_under_100k():
    min_p, max_p, query = extract_price_info("cơm tấm dưới 100k")
    assert max_p == 100000.0
    assert "cơm tấm" in query

def test_extract_price_info_under_200k():
    min_p, max_p, query = extract_price_info("trà sữa dưới 200k")
    assert max_p == 200000.0
    assert "trà sữa" in query

def test_extract_price_info_under_300k():
    min_p, max_p, query = extract_price_info("món lẩu dưới 300k")
    assert max_p == 300000.0
    assert "món lẩu" in query

def test_extract_price_info_under_400k():
    min_p, max_p, query = extract_price_info("combo ăn gia đình dưới 400k")
    assert max_p == 400000.0
    assert "combo" in query

def test_extract_price_info_under_500k():
    min_p, max_p, query = extract_price_info("bàn tiệc hải sản dưới 500k")
    assert max_p == 500000.0

def test_extract_price_info_range():
    min_p, max_p, query = extract_price_info("cho 2 tô phở bò tầm 45k đến 50k nhé")
    assert min_p == 45000.0
    assert max_p == 50000.0

def test_extract_food_query_conversational():
    q = extract_food_query("Tôi muốn ăn 2 phần bún chả nhé ạ")
    assert "bún chả" in q
