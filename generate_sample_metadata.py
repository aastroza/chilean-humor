#!/usr/bin/env python3
"""
Generate sample metadata for testing the curation interface.

This script creates a small sample of clips_metadata.json for testing purposes,
without requiring the full clip processing pipeline.
"""

import json
import os


def generate_sample_metadata(output_file='./clips_metadata_sample.json', num_samples=10):
    """
    Generate sample metadata entries.

    Args:
        output_file: Path to output JSON file
        num_samples: Number of sample entries to create
    """
    sample_transcripts = [
        "Por favor, gracias a ella, de inmediato a continuación...",
        "El humor chileno tiene una larga historia en nuestro país.",
        "Buenas noches damas y caballeros, bienvenidos al show.",
        "Hoy vamos a hablar sobre las cosas que nos pasan todos los días.",
        "¿Se acuerdan cuando éramos niños y jugábamos en la calle?",
        "La política en Chile es como un circo, pero sin los payasos... bueno, con algunos payasos.",
        "Me encanta venir al Festival de Viña del Mar cada año.",
        "El transporte público en Santiago es toda una aventura.",
        "Las abuelas chilenas son las mejores del mundo entero.",
        "Nosotros los chilenos tenemos un acento muy particular.",
        "El fútbol nos une a todos los chilenos, para bien o para mal.",
        "La comida chilena es deliciosa, especialmente los completos.",
        "En Chile tenemos una relación especial con los terremotos.",
        "Las playas de Chile son hermosas pero el agua está helada.",
        "Santiago tiene muchos cerros, perfectos para hacer ejercicio.",
        "El español chileno tiene tantos modismos que nadie más entiende.",
        "Los veranos en Chile son calurosos y los inviernos fríos.",
        "La cordillera de los Andes es majestuosa y siempre visible.",
        "En septiembre celebramos las fiestas patrias con mucha energía.",
        "El vino chileno es reconocido mundialmente por su calidad."
    ]

    metadata = []

    for i in range(num_samples):
        metadata.append({
            'audio_path': f'./audios/sample_routine_{i % 3 + 1}_clip_{i+1:06d}.mp3',
            'transcript': sample_transcripts[i % len(sample_transcripts)],
            'routine_id': (i % 3) + 1,
            'start_time': float(i * 30),
            'end_time': float((i * 30) + 25),
            'duration': 25.0,
            'selected': True
        })

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ Generated {num_samples} sample entries")
    print(f"✓ Saved to: {output_file}")
    print(f"\nNote: Audio files don't exist - this is for interface testing only.")
    print(f"To use with curate_clips.py, edit the METADATA_FILE constant.")


if __name__ == '__main__':
    generate_sample_metadata()
