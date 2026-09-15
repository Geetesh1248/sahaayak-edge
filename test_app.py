import unittest
from unittest.mock import patch, MagicMock
import app

class TestSahaayakEdge(unittest.TestCase):
    def setUp(self):
        # Reset global state before each test
        app.faiss_index = None
        app.chunk_data = []

    def test_missing_pdf(self):
        """Test handling of missing/None PDF path"""
        res = app.process_pdf(None)
        self.assertIn("Error: Please upload a valid PDF", res)

    @patch('app.os.path.exists')
    @patch('app.pdfplumber.open')
    def test_empty_pdf(self, mock_pdf_open, mock_exists):
        """Test handling of a PDF with no pages"""
        mock_exists.return_value = True
        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        res = app.process_pdf("dummy.pdf")
        self.assertIn("Error: The uploaded PDF has no pages", res)
        
    @patch('app.os.path.exists')
    @patch('app.pdfplumber.open')
    def test_image_only_pdf(self, mock_pdf_open, mock_exists):
        """Test handling of a PDF with pages but no text"""
        mock_exists.return_value = True
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "" # No text
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        res = app.process_pdf("dummy.pdf")
        self.assertIn("Error: No text could be extracted", res)

    @patch('app.os.path.exists')
    @patch('app.pdfplumber.open')
    def test_pdf_extraction_and_chunking(self, mock_pdf_open, mock_exists):
        """Test successful PDF extraction and chunking"""
        mock_exists.return_value = True
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        # Provide enough text to create at least one chunk
        mock_page.extract_text.return_value = "This is a test document with some text to ensure chunking works correctly. " * 10
        mock_pdf.pages = [mock_page]
        mock_pdf_open.return_value.__enter__.return_value = mock_pdf
        
        res = app.process_pdf("dummy.pdf")
        self.assertIn("PDF processed successfully", res)
        self.assertTrue(len(app.chunk_data) > 0)
        
    def test_low_confidence_fallback(self):
        """Test fallback when retrieval distance is too high"""
        app.chunk_data = [{"text": "dummy context", "page": 1}]
        app.faiss_index = MagicMock()
        # Mock search to return a high distance (low confidence) -> greater than CONFIDENCE_THRESHOLD (1.5)
        app.faiss_index.search.return_value = ([[2.0]], [[0]])
        
        answer, source = app.answer_question("What is X?")
        self.assertEqual(answer, "I'm not confident about this — please verify with your instructor")
        self.assertEqual(source, "Source: N/A (Low Confidence)")
        
    @patch('app.ollama.generate')
    def test_citation_formatting_and_missing_ollama(self, mock_generate):
        """Test citation formatting and Ollama connection error handling"""
        app.chunk_data = [{"text": "dummy", "page": 2}, {"text": "dummy2", "page": 3}]
        app.faiss_index = MagicMock()
        # Mock search to return a low distance (high confidence)
        app.faiss_index.search.return_value = ([[0.5]], [[0, 1, 0]])
        
        # Test 1: Successful generation for citation formatting
        mock_generate.return_value = {'response': 'Here is the generated answer.'}
        answer, source = app.answer_question("Test?")
        self.assertEqual(answer, "Here is the generated answer.")
        self.assertIn("Source: Page(s) 2, 3", source)
        
        # Test 2: Missing Ollama exception handling
        mock_generate.side_effect = Exception("Connection refused")
        answer, source = app.answer_question("Test?")
        self.assertIn("Error: Could not connect to Ollama", answer)

if __name__ == '__main__':
    unittest.main()
