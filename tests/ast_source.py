import ast
import re


def node_source(source, node):
    """Return a stable source representation on every supported Python version."""
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError("AST node has no source segment")
    segment = segment.replace('"', "'")
    compact = " ".join(segment.split())
    compact = re.sub(r"([([{])\s+", r"\1", compact)
    compact = re.sub(r"\s+([])},:])", r"\1", compact)
    compact = re.sub(r"(?<=\d)e([+-])(\d)(?!\d)", r"e\g<1>0\2", compact)
    return segment + "\n" + compact
