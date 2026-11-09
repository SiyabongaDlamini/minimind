import zipfile

import pytest

from minimind import KnowledgeBase
from minimind.knowledge import chunk_text, read_document, tokenize

POLICY = """
Employees must give 30 days of written notice before resigning. The notice period
may be shortened by agreement with a manager.

Annual leave is 25 days per year. Unused leave can be carried over for one year.

Expense reports are due on the fifth of each month and are paid within two weeks.
"""


@pytest.fixture
def kb():
    base = KnowledgeBase(max_words=25)
    base.add_text(POLICY, source="policy.txt")
    return base


def test_answers_from_the_right_sentence(kb):
    answer = kb.ask("How many days of notice before resigning?")
    assert answer.found
    assert "30 days" in answer.text
    assert "policy.txt" in str(answer)


def test_different_questions_find_different_passages(kb):
    assert "25 days" in kb.ask("how much annual leave do I get").text
    assert "fifth" in kb.ask("when are expense reports due").text


def test_unrelated_question_is_declined_not_invented(kb):
    answer = kb.ask("What is the capital of Mongolia?")
    assert not answer.found
    assert "couldn't find" in answer.text


def test_empty_knowledge_base():
    assert not KnowledgeBase().ask("anything").found


def test_reads_txt_docx_and_skips_unsupported(tmp_path):
    (tmp_path / "a.txt").write_text("The server password rotates every 90 days.")
    docx = tmp_path / "b.docx"
    body = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Backups run nightly at 02:00.</w:t></w:r></w:p></w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", body)
    (tmp_path / "c.png").write_bytes(b"not text")

    assert "Backups run nightly" in read_document(docx)
    base = KnowledgeBase()
    report = base.add_paths([tmp_path])
    assert report["a.txt"] > 0 and report["b.docx"] > 0 and report["c.png"] == -1
    assert "02:00" in base.ask("when do backups run").text


def test_unsupported_file_message_mentions_ocr(tmp_path):
    img = tmp_path / "scan.png"
    img.write_bytes(b"x")
    with pytest.raises(ValueError, match="OCR"):
        read_document(img)


def test_save_and_load(kb, tmp_path):
    kb.save(tmp_path / "kb.json")
    loaded = KnowledgeBase.load(tmp_path / "kb.json")
    assert len(loaded) == len(kb)
    assert "30 days" in loaded.ask("notice before resigning").text


def test_markdown_heading_is_not_glued_to_the_answer():
    base = KnowledgeBase()
    base.add_text("# Handbook\nStaff must give 30 days of notice before resigning.", "h.md")
    answer = base.ask("notice before resigning")
    assert answer.text == "Staff must give 30 days of notice before resigning."


def test_chunking_and_tokenizing():
    chunks = chunk_text("One two three. " * 40, max_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 24 for c in chunks)
    assert tokenize("The cats are running") == ["cat", "runn"]
