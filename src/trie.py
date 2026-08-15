class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.token_id: int | None = None


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_id: int, token_str: str) -> None:
        """Insert a token into the trie.

        Args:
            token_id: The ID of the token to insert.
            token_str: The string representation of the token.
        """
        node = self.root
        for char in token_str:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token_id = token_id

    def build_from_vocab(self, vocab: dict[str, int]) -> None:
        """Insert every token from a vocab dict into the trie.

        Args:
            vocab: Mapping of token strings to token IDs.
        """
        for token_str, token_id in vocab.items():
            self.insert(token_id, token_str)

    def find_prefix(self, prefix: str) -> TrieNode | None:
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node

    def collect_token_ids(self, node: TrieNode) -> list[int]:
        result = []
        stack = [node]

        while stack:
            current = stack.pop()

            if current.token_id is not None:
                result.append(current.token_id)

            for child in current.children.values():
                stack.append(child)

        return result

    def search_prefix(self, prefix: str) -> list[int]:
        """Find all token IDs in the trie whose string starts with the given prefix.

        Args:
            prefix: The string that matching tokens must start with.

        Returns:
            A list of token IDs.
        """
        node = self.find_prefix(prefix)
        if node is None:
            return []
        return self.collect_token_ids(node)
