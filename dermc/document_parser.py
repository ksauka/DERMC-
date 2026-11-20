"""
Document parsing module that handles:
1. Hierarchical markdown structure extraction
2. Section-based content segmentation
3. Hierarchy path tracking (up to max_depth levels)
4. Flat vs hierarchical parsing modes
"""
import json
import pandas as pd
import logging
from tqdm import tqdm
from typing import Dict

logger = logging.getLogger(__name__)

class DocumentParser:
    """Optimized document parser with hierarchy preservation"""
    def __init__(self, max_depth: int = 4):
        """
        Initialize DocumentParser.
        Args:
            max_depth: Maximum depth of hierarchy to parse.
        """
        self.max_depth = max_depth
        self.reset_parser_state()

    def reset_parser_state(self) -> None:
        """
        Reset the parser's internal state for a new document.
        """
        from typing import Optional
        self.structure = {'hierarchy_tree': {}, 'content_with_hierarchy': []}
        self.current_levels: list[Optional[str]] = [None] * self.max_depth
        self.content_buffer = []

    def clean_line(self, line: str) -> str:
        """
        Remove markdown/list prefixes from a line.
        Args:
            line: The input line.
        Returns:
            Cleaned line string.
        """
        patterns = {
            '# ': (2,), '## ': (3,), '### ': (4,), '#### ': (5,),
            '* ': (2,), '- ': (2,), '+ ': (2,)
        }
        for prefix, (strip_len,) in patterns.items():
            if line.startswith(prefix):
                return line[strip_len:].strip()
        return line.strip()

    def process_line(self, line: str) -> None:
        """
        Process a single line of text, updating hierarchy/content buffers.
        Args:
            line: The input line.
        """
        clean_line = self.clean_line(line)
        # Build hierarchy path from current levels (e.g., "Section/Subsection")
        hierarchy_path = "/".join(filter(None, self.current_levels))

        # Route line to appropriate handler based on markdown syntax
        if line.startswith('#'):
            depth = min(line.count('#'), self.max_depth)
            self._process_header(clean_line, depth, hierarchy_path)
        elif line.startswith(('* ', '- ', '+ ')):
            self._process_list_item(clean_line, hierarchy_path)
        elif clean_line:
            self.content_buffer.append(clean_line)

    def _process_header(self, header: str, depth: int, hierarchy_path: str) -> None:
        """
        Process a header line, updating hierarchy and flushing content buffer.
        Args:
            header: Header text.
            depth: Header depth (level).
            hierarchy_path: Current hierarchy path string.
        """
        # Clear deeper levels when entering new section at this depth
        for i in range(depth, self.max_depth):
            self.current_levels[i] = None
        self.current_levels[depth-1] = header
        # Flush accumulated content before starting new section
        if self.content_buffer:
            self.flush_content_buffer()
        current = self.structure['hierarchy_tree']
        for level in self.current_levels[:depth]:
            if level:
                current = current.setdefault(level, {})
        self.structure['content_with_hierarchy'].append({
            'text': header,
            'hierarchy': hierarchy_path,
            'is_header': True,
            'level': depth,
            'parsing_mode': 'hierarchical'
        })

    def _process_list_item(self, item: str, hierarchy_path: str) -> None:
        """
        Process a list item line, updating hierarchy tree and content.
        Args:
            item: List item text.
            hierarchy_path: Current hierarchy path string.
        """
        if self.content_buffer:
            self.flush_content_buffer()
        self.structure['content_with_hierarchy'].append({
            'text': item,
            'hierarchy': hierarchy_path,
            'is_attribute': True,
            'parsing_mode': 'hierarchical'
        })
        current = self.structure['hierarchy_tree']
        for level in filter(None, self.current_levels):
            current = current[level]
        current[item] = None

    def flush_content_buffer(self) -> None:
        """
        Flush the content buffer, adding its contents to the structure.
        """
        if not self.content_buffer:
            return
        content_text = " ".join(self.content_buffer)
        if content_text:
            self.structure['content_with_hierarchy'].append({
                'text': content_text,
                'hierarchy': "/".join(filter(None, self.current_levels)),
                'is_content': True,
                'parsing_mode': 'hierarchical'
            })
        self.content_buffer = []

    def parse_document(self, document_text: str) -> Dict:
        """
        Parse a single document string into a hierarchy/content structure.
        Args:
            document_text: The document text to parse.
        Returns:
            Parsed structure as a dict.
        """
        self.reset_parser_state()
        if pd.isna(document_text) or document_text is None:
            document_text = ""
        else:
            document_text = str(document_text)
        [self.process_line(line.strip()) for line in document_text.split('\n') if line.strip()]
        self.flush_content_buffer()
        return self.structure

    def parse_dataframe(self, df: pd.DataFrame, text_column: str = 'Document') -> pd.DataFrame:
        """
        Parse a DataFrame of documents, returning a DataFrame with parsed structures.
        Args:
            df: Input DataFrame.
            text_column: Name of the column containing document text.
        Returns:
            DataFrame with parsed hierarchy/content columns.
        """
        parsed_data = []
        logger.info("Parsing documents hierarchy...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing documents (hierarchical)"):
            doc_structure = self.parse_document(row[text_column])
            parsed_data.append({
                'Doc ID': row['Doc ID'],
                'Answer ID': row['Answer ID'],
                'hierarchy_tree': json.dumps(doc_structure['hierarchy_tree']),
                'content_with_hierarchy': json.dumps(doc_structure['content_with_hierarchy']),
                'hierarchy_context': self._get_hierarchy_context(doc_structure),
                'parsing_mode': 'hierarchical'
            })
            self.reset_parser_state()
        return pd.DataFrame(parsed_data)

    def _get_hierarchy_context(self, structure: Dict) -> str:
        """
        Build a string representation of the document's hierarchy tree.
        Args:
            structure: Parsed document structure dict.
        Returns:
            String representation of the hierarchy tree.
        """
        context_lines = ["Document Hierarchy:"]
        def build_tree(node, depth=0):
            for key, value in node.items():
                context_lines.append(f"{'  ' * depth}- {key}")
                if isinstance(value, dict):
                    build_tree(value, depth+1)
        build_tree(structure['hierarchy_tree'])
        return "\n".join(context_lines)
