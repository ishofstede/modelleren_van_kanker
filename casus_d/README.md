# Tumor Objectdetectie met YOLO

Dit project richt zich op het trainen van een neuraal netwerk met behulp van het YOLO (You Only Look Once) objectdetectie-algoritme om tumoren te detecteren en te lokaliseren in medische afbeeldingen.

## Inhoudsopgave

**Overzicht**
**Installatie**
**Dataset Voorbereiding**


## Overzicht

In dit project wordt een YOLO-model ontwikkeld voor de detectie en lokalisatie van tumoren in medische afbeeldingen. De workflow omvat het voorbereiden van de dataset, het trainen van het model, en het maken van voorspellingen voor nieuwe afbeeldingen.

## Installatie

Om dit project uit te voeren, zijn enkele vereisten nodig:

- Python 3.x
- `ultralytics` bibliotheek voor YOLO (installatie: `pip install ultralytics`)
- `PIL` bibliotheek voor afbeeldingsverwerking (installatie: `pip install pillow`)

## Dataset Voorbereiding

De dataset wordt voorbereid door annotaties om te zetten naar het YOLO-formaat. Hierbij worden de coördinaten van bounding boxes genormaliseerd en opgeslagen in bijbehorende tekstbestanden. De split in trainings- en validatiesets wordt georganiseerd in een specifieke mapstructuur geschikt voor YOLO. Er zijn een aantal paden aangegeven bij elke functie, pas deze aan aan je eigen directory structure. De Directories waar de training data in terrecht komt worden zelf aangemaakt.
