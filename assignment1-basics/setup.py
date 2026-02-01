from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension(
        name="cs336_basics.BPE_Tokenizer.bpe_cpython._merge_pair_and_count_pair_difference",
        sources=[
            "cs336_basics/BPE_Tokenizer/bpe_cpython/_merge_pair_and_count_pair_difference.pyx"
        ],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="cs336_basics",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)