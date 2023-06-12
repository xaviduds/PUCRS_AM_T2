# Goals

- [x] Load model
- [x] Train model
- [ ] Load libraries for Transfer Learning
- [ ] Edit CNN for the problem
- [ ] Transfer Learn for correct liquid/empty bottle segmentation
- [ ] Calculate the bottle's 'fullness'

# Observações

Eduardo: quando faço segmentação com modelo nano (n) fica uma parte da garrafa de fora, se formos analisar a 'fullness' dela esses pixels não vão ser contados. Podemos usar um modelo maior que o nano mas vai ficar mais pesado, lento. To rodando localmente, pode não haver esse problema no Colab.
Eduardo: vamos usar as fotos segmentadas pelo YOLOv8n-seg.pt. Pegamos a binary mask dessa foto e usamos 50 Linhas De Python pra computar dentre os pixels da garrafa os pixels de líquido.

# Estrutura de arquivos e folders:

O README.txt precisamos entregar pro sor segundo o enunciado do T2 em README.md.
Dentro de '3 - code' tem o folder 'runs', criado pelo YOLOv8. As segmentações podem ser encontradas em 'runs/segment/predict'.
