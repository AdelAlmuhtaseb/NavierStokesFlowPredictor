| Model | velocity u error | velocity v error | pressure Cp error | ms/sample | Parameters | Training samples |
|---|---|---|---|---|---|---|
| U-Net | 0.005 | 0.055 | 0.059 | 0.9 | 482,067 | 1200 |
| FNO | 0.042 | 0.399 | 0.439 | 1.5 | 1,186,371 | 1200 |

Panel solver: 31 ms per sample on this computer.
U-Net is about 35x faster than the solver.
FNO is about 21x faster than the solver.
