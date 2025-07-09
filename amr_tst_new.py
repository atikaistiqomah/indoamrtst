import penman
from amr_to_text import AMRToTextBase
from dataclasses import dataclass
from text_to_amr import TextToAMR
from utils import make_no_metadata_graph
from tqdm import tqdm
from typing import Optional

@dataclass
class AMRTSTDetailedResult:
    source_text: list[str]
    source_amr: list[penman.Graph]
    target_amr: list[penman.Graph]
    target_text: list[str]

    def to_list(self) -> list[dict[str, str | list[str]]]:
        expected_length = len(self.source_text)
        assert len(self.source_amr) == expected_length, f"Expecting {expected_length}, got {len(self.source_amr)}"
        assert len(self.target_amr) == expected_length, f"Expecting {expected_length}, got {len(self.target_amr)}"
        assert len(self.target_text) == expected_length, f"Expecting {expected_length}, got {len(self.target_text)}"

        data = []
        for i in range(expected_length):
            sa = make_no_metadata_graph(self.source_amr[i])
            ta = make_no_metadata_graph(self.target_amr[i])

            data.append({
                "source_text": self.source_text[i],
                "source_amr": penman.encode(sa, indent=None),
                "target_amr": penman.encode(ta, indent=None),
                "target_text": self.target_text[i],
            })

        return data

class AMRTST:
    """
    AMR-TST tanpa Style Detector dan Style Rewriting.
    """
    def __init__(self, t2a: TextToAMR, a2t: AMRToTextBase):
        self.t2a = t2a
        self.a2t = a2t
        self.last_source_graphs = []
        self.last_target_texts = []

    def __call__(self, texts: list[str]):
        """
        Pipeline AMR-TST: Mengonversi teks ke AMR dan kembali ke teks.
        """
        graphs = self.t2a(texts)  # AMR Parsing
        self.last_source_graphs = graphs
        
        # AMR Generation
        try:
            target_texts = self.a2t(graphs)
        except Exception as e:
            print(f"Warning: Can't process AMR graphs for generation.\nError: {e}")
            target_texts = ["" for _ in graphs]
        
        self.last_target_texts = target_texts

        return target_texts, AMRTSTDetailedResult(
            source_text=texts,
            source_amr=graphs,
            target_amr=graphs,  # Tidak ada rewriting, jadi target = source
            target_text=target_texts
        )
