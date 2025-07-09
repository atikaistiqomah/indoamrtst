import json
import sys

from utils import to_amr_with_pointer

SENT_PREFIX = "# ::snt "
def sent_amr_iter(path: str):
    sent_offset = len(SENT_PREFIX)

    with open(path) as fp:
        status = "find_non_empty_line"
        current_amr = ""
        current_sent = ""

        for line in fp:
            line = line.strip()
            match status:
                case "find_non_empty_line":
                    if line == "":
                        continue

                    if line.startswith("("):
                        current_sent = ""
                        current_amr = line
                        status = "select_amr_until_blank_line"
                    else:
                        if line.startswith(SENT_PREFIX):
                            current_sent = line[sent_offset:].strip()
                        status = "find_end_of_header"

                case "find_end_of_header":
                    if line.startswith("("):
                        current_amr = line
                        status = "select_amr_until_blank_line"
                    elif line == "":
                        status = "find_begin_of_amr"
                    elif line.startswith(SENT_PREFIX):
                        if current_sent != "":
                            yield current_sent, "", "", "AMR is empty"
                        current_sent = line[sent_offset:].strip()
                    # else: ignore

                case "find_begin_of_amr":
                    if line == "":
                        continue
                    
                    if line.startswith("("):
                        current_amr = line
                        status = "select_amr_until_blank_line"
                    else:
                        yield current_sent, "", "", "AMR is empty"
                        current_sent = ""
                        status = "find_end_of_header"
                    
                case "select_amr_until_blank_line":
                    if line == "" or line.startswith("#"):
                        try:
                            amr_with_pointer = to_amr_with_pointer(current_amr)
                            if current_sent != "":
                                yield current_sent, current_amr, amr_with_pointer, ""
                            else:
                                yield current_sent, current_amr, amr_with_pointer, "Sentence is empty"
                        except ValueError as e:
                            yield current_sent, current_amr, "", str(e)
                    
                        current_sent = ""
                        current_amr = ""
                        if line.startswith("#"):
                            status = "find_end_of_header"
                        else:
                            status = "find_non_empty_line"

                    elif current_amr == "":
                        current_amr = line
                        
                    else:
                        current_amr += " " + line
            
        if status == "select_amr_until_blank_line":
            try:
                amr_with_pointer = to_amr_with_pointer(current_amr)
                if current_sent != "":
                    yield current_sent, current_amr, amr_with_pointer, ""
                else:
                    yield current_sent, current_amr, amr_with_pointer, "Sentence is empty"
            except ValueError as e:
                yield current_sent, current_amr, "", str(e)

def to_jsonl_dataset_2(input_path: str, output_path: str):
    error_count = 0
    with open(output_path, mode="w") as fp_out:
        for sent, raw_amr, amr_with_pointer, error_message in sent_amr_iter(input_path):
            if error_message == "":
                print(
                    json.dumps({"sent": sent, "amr": amr_with_pointer, "lang": "id"}),
                    file=fp_out
                )
            else:
                error_count += 1
                print(f"(Error {error_count}) {error_message}")
                print(f"Sentence: {sent}")
                print(f"AMR:")
                print(raw_amr)
                print("---")

    if error_count > 0:
        print(f"{error_count=}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError(f"Expected command format: {sys.argv[0]} <input-path> <output-path>")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    to_jsonl_dataset_2(input_path, output_path)
