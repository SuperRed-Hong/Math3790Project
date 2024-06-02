import transformers


def load_tokenizer(model_name: str = "bert-base-chinese"):
    """Load the tokenizer for your model and add some special tokens.

    Args:
        model_name (str): The model name which used as a base model for CWS. \
            The model name need to find in Hugging Face.

    Returns:
        transformers.BertTokenizerFast : A pretrained tokenizer.
    """

    return transformers.BertTokenizerFast.from_pretrained(
        # model_name,
        '../../bert-base-chinese',
        pad_token="[PAD]",
        additional_special_tokens=[
            "[AS]",
            "[CIT]",
            "[MSR]",
            "[PKU]",
            "[UNC]",
            "[CTB6]",
            "[CNC]",
            "[SXU]",
            "[UD]",
            "[WTB]",
            "[ZX]",
        ],
    )

if __name__ == "__main__":
    text = "新中国建立初期,面临的国际形势异常严峻:以美国为首的帝国主义敌视新中国,企图以政治孤立、经济封锁和军事威胁把新中国扼杀在摇篮里。当时中国外交的中心任务是:巩固新生的无产阶级政权,为社会主义和平建设争取一个有利的国际环境。为此,毛泽东确立了执行和平外交政策三大基本方针,即“另起炉灶”、“一边倒”和“打扫干净屋子再请客”。公开宣布站在社会主义一边,坚决反对美国的侵略政策和战争政策,彻底清除帝国主义在华特权和势力。"
    tokenizer=load_tokenizer()
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    tokenized_str = tokenizer(
        [x[0] for x in data],
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )
    print(tokens)
