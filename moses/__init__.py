"""Statistical machine translation using Moses."""

from itertools import chain

from sparv.api import Annotation, Binary, Config, Output, annotator, util

TOKEN_SEP = "\n"
CHAR_SEP = " "


@annotator(
    "Statistical machine translation using Moses",
    config=[
        Config("moses.bin", description="Absolute path to Moses executable"),
        Config("moses.ini", description="Absolute path to Moses ini file"),
        Config("moses.threads", default=12, description="Number of threads to use"),
    ],
)
def run(
    word: Annotation = Annotation("<token:word>"),
    sentence: Annotation = Annotation("<sentence>"),
    out: Output = Output("<token>:moses.word"),
    executable: Binary = Binary("[moses.bin]"),
    config: str = Config("moses.ini"),
    threads: int = Config("moses.threads"),
) -> None:
    """Annotate using Moses."""
    sentences, orphans = sentence.get_children(word)
    sentences.append(orphans)
    word_list = list(word.read())
    out_annotation = word.create_empty_attribute()

    stdin = TOKEN_SEP.join(
        CHAR_SEP.join(word_list[token_index])
        for sent in sentences
        for token_index in sent
    )

    args = ["--threads", threads, "-v", "0", "-f", config]
    stdout, _ = util.system.call_binary(executable, args, stdin, encoding="utf-8")

    for token_index, moses_token in zip(chain(*sentences), stdout.split(TOKEN_SEP)):
        out_annotation[token_index] = "".join(moses_token.split(CHAR_SEP))

    out.write(out_annotation)
