class Template:
    def encode(self, sample):
        """
        Return prompted version of the example (without the answer/candidate)
        """
        raise NotImplementedError

    def verbalize(self, sample, candidate):
        """
        Return the prompted version of the example (with the answer/candidate)
        """
        return candidate

    def encode_sfc(self, sample):
        """
        Same as encode, but for SFC (calibration) -- this usually means the input is not included
        """
        return "<mask>"

    def verbalize_sfc(self, sample, candidate):
        """
        Same as verbalize, but for SFC (calibration) -- this usually means the input is not included
        """
        return candidate


class DROPTemplate(Template):

    def encode(self, sample):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question'].strip()
        # title = sample.data['title']
        context = sample.data['context']
        answer = sample.data['answers'][0]  # there are multiple answers. for the prompt we only take the first one

        return f"Passage: {context}\nQuestion: {question}\nAnswer: {answer}\n"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class MATH10KTemplate(Template):
    def encode(self, sample):
        question = sample.data['question']
        return f"{question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question']
        output = sample.data['output']
        return f"{question}\nAnswer: {output}"

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError


class COMMONSENSE170KTemplate(Template):
    def encode(self, sample):
        question = sample.data['question']
        return f"{question}\nAnswer:"

    def verbalize(self, sample, candidate):
        question = sample.data['question']
        return f"{question}\nAnswer: the correct answer is {candidate}."

    def encode_sfc(self, sample):
        raise NotImplementedError

    def verbalize_sfc(self, sample, candidate):
        raise NotImplementedError
