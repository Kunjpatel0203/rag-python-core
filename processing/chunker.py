# # from langchain_text_splitters import RecursiveCharacterTextSplitter


# # def chunk_text(text: str):
# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=800,
# #         chunk_overlap=100
# #     )
# #     return splitter.split_text(text)








# import re
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# SECTION_REGEX = re.compile(
#     r"""
#     ^(
#         [A-Z][A-Z\s\-0-9()]{5,} |          # ALL CAPS headings
#         \d+\.\s+[A-Z].+ |                  # Numbered headings
#         [A-Z][a-z].+?:$                    # Title-like headings ending with :
#     )
#     """,
#     re.VERBOSE
# )


# def _extract_sections(text: str):
#     """
#     Splits text into (section_title, section_text).
#     If no heading is found, assigns to 'GENERAL'.
#     """
#     lines = text.splitlines()

#     sections = []
#     current_title = "GENERAL"
#     current_buffer = []

#     for line in lines:
#         clean = line.strip()

#         if SECTION_REGEX.match(clean):
#             # Save previous section
#             if current_buffer:
#                 sections.append(
#                     (current_title, "\n".join(current_buffer))
#                 )
#                 current_buffer = []

#             current_title = clean
#         else:
#             if clean:
#                 current_buffer.append(clean)

#     if current_buffer:
#         sections.append(
#             (current_title, "\n".join(current_buffer))
#         )

#     return sections


# def chunk_text(text: str):
#     """
#     Section-aware chunking:
#     - Preserves headings
#     - Prevents cross-topic mixing
#     - Fully backward compatible
#     """

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=100
#     )

#     chunks = []
#     sections = _extract_sections(text)

#     for section_title, section_text in sections:
#         section_chunks = splitter.split_text(section_text)

#         for chunk in section_chunks:
#             enriched_chunk = (
#                 f"SECTION: {section_title}\n"
#                 f"{chunk}"
#             )
#             chunks.append(enriched_chunk)

#     return chunks










from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_text(text)
