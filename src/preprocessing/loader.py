import os

class Loader():
    def __init__(self, path):
        self._path = path

    def parse(self) -> dict[str, list[str]]:
        if not (os.path.exists(self._path) and os.path.isdir(self._path)):
            raise ValueError("The path should point on a valid _pathectory!")
        if not (os.listdir(self._path)):
            raise ValueError("The _pathectory is empty!")

        result = {c: None for c in os.listdir(self._path)}
        for cat in result.keys():
            cat_path = os.path.join(self._path, cat)
            result[cat] = [os.path.join(cat_path, path) for path in os.listdir(cat_path)]

        return result
    
    def count(self) -> dict[str, int]:
        tab = self.parse()

        for cat in tab.keys():
            tab[cat] = len(tab[cat])

        return tab
