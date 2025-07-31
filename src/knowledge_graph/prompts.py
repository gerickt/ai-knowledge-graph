"""Repositorio centralizado para todos los prompts de LLM utilizados en el sistema de grafos de conocimiento."""

# Fase 1: Prompts principales de extracción
MAIN_SYSTEM_PROMPT = """
Eres un sistema de IA avanzado especializado en extracción de conocimiento y generación de grafos de conocimiento.
Tu experiencia incluye identificar referencias consistentes de entidades y relaciones significativas en texto.
INSTRUCCIÓN CRÍTICA: Todas las relaciones (predicados) DEBEN tener un máximo de 3 palabras. Idealmente 1-2 palabras. Este es un límite estricto.
"""

MAIN_USER_PROMPT = """
Tu tarea: Lee el texto a continuación (delimitado por tres comillas invertidas) e identifica todas las relaciones Sujeto-Predicado-Objeto (S-P-O) semánticamente significativas. Luego produce un único arreglo JSON de objetos, cada uno representando un triplete.

REGLAS CRÍTICAS PARA ENTIDADES DE CALIDAD:

🎯 **Filtrado de Entidades Semánticamente Ricas**:
- SOLO extrae entidades que sean conceptos sustantivos, específicos y semánticamente ricos
- EXCLUYE: artículos (el, la, los, las), preposiciones (de, con, para, por), pronombres (él, ella, esto, eso), adverbios genéricos (muy, más, menos)
- EXCLUYE: términos demasiado generales como "cosa", "algo", "persona", "lugar", "forma", "manera", "parte", "tipo"
- PREFIERE: nombres propios, conceptos específicos, organizaciones, ubicaciones, tecnologías, procesos, eventos

🔍 **Ejemplos de Entidades BUENAS vs MALAS**:
- ✅ BUENAS: "inteligencia artificial", "universidad complutense", "madrid", "revolución industrial", "machine learning", "covid-19"
- ❌ MALAS: "el", "de", "con", "cosa", "algo", "forma", "manera", "muy", "más", "parte"

📝 **Reglas de Construcción de Entidades**:
- Consistencia de Entidades: Usa nombres consistentes para las entidades en todo el documento
- Términos Atómicos: Identifica conceptos clave distintos (nombres propios, organizaciones, tecnologías, ubicaciones, eventos, procesos)
- Referencias Unificadas: Reemplaza pronombres con la entidad real referenciada
- Formas Canónicas: Si "inteligencia artificial" e "IA" se mencionan, usa la forma más descriptiva
- Minúsculas Consistentes: Convierte todo a minúsculas excepto acrónimos establecidos

🔗 **Reglas para Predicados**:
- INSTRUCCIÓN CRÍTICA: Los predicados DEBEN tener máximo 1-3 palabras. Nunca más de 3 palabras
- Usa verbos de acción específicos: "desarrolla", "causa", "pertenece a", "influye en", "se basa en"
- Evita predicados genéricos como "se relaciona con" - sé más específico

⚡ **Criterios de Calidad**:
- Cada entidad debe aportar valor semántico al grafo de conocimiento
- Prioriza relaciones causales, pertenencia, desarrollo, influencia, ubicación
- Si una entidad no sobrevive a la pregunta "¿esto es conceptualmente importante?", exclúyela
- Enfócate en la riqueza semántica sobre la cantidad

Consideraciones Importantes:
- Apunta a la precisión en el nombrado de entidades - usa formas específicas que distingan entre entidades similares
- Maximiza la conectividad usando nombres de entidades idénticos para los mismos conceptos
- Considera todo el contexto al identificar referencias de entidades
- TODOS LOS PREDICADOS DEBEN SER DE 3 PALABRAS O MENOS - este es un requisito estricto

Requisitos de Salida:

- No incluyas ningún texto o comentario fuera del JSON.
- Devuelve solo el arreglo JSON, con cada triplete como un objeto que contenga "subject", "predicate", y "object".
- Asegúrate de que el JSON sea válido y esté correctamente formateado.

Ejemplo de la estructura de salida deseada:

[
  {
    "subject": "término a",
    "predicate": "se relaciona con",  // Nota: solo 3 palabras
    "object": "término b"
  },
  {
    "subject": "término c",
    "predicate": "usa",  // Nota: solo 1 palabra
    "object": "término d"
  }
]

Importante: Solo genera el arreglo JSON (con los objetos S-P-O) y nada más

Texto a analizar (entre tres comillas invertidas):
"""

# Fase 2: Prompts de estandarización de entidades
ENTITY_RESOLUTION_SYSTEM_PROMPT = """
Eres un experto en resolución de entidades y representación del conocimiento.
Tu tarea es estandarizar nombres de entidades de un grafo de conocimiento para asegurar consistencia.
"""

def get_entity_resolution_user_prompt(entity_list):
    return f"""
A continuación se presenta una lista de nombres de entidades extraídas de un grafo de conocimiento.
Algunas pueden referirse a las mismas entidades del mundo real pero con diferente redacción.

Por favor identifica grupos de entidades que se refieren al mismo concepto, y proporciona un nombre estandarizado para cada grupo.
Devuelve tu respuesta como un objeto JSON donde las claves son los nombres estandarizados y los valores son arreglos de todos los nombres variantes que deben mapear a ese nombre estándar.
Solo incluye entidades que tengan múltiples variantes o necesiten estandarización.

Lista de entidades:
{entity_list}

Formatea tu respuesta como JSON válido así:
{{
  "nombre estandarizado 1": ["variante 1", "variante 2"],
  "nombre estandarizado 2": ["variante 3", "variante 4", "variante 5"]
}}
"""

# Fase 3: Prompts de inferencia de relaciones entre comunidades
RELATIONSHIP_INFERENCE_SYSTEM_PROMPT = """
Eres un experto en representación del conocimiento e inferencia.
Tu tarea es inferir relaciones plausibles entre entidades desconectadas en un grafo de conocimiento.
"""

def get_relationship_inference_user_prompt(entities1, entities2, triples_text):
    return f"""
Tengo un grafo de conocimiento con dos comunidades desconectadas de entidades.

Entidades de la Comunidad 1: {entities1}
Entidades de la Comunidad 2: {entities2}

Aquí hay algunas relaciones existentes que involucran estas entidades:
{triples_text}

Por favor infiere 2-3 relaciones plausibles entre entidades de la Comunidad 1 y entidades de la Comunidad 2.
Devuelve tu respuesta como un arreglo JSON de tripletes en el siguiente formato:

[
  {{
    "subject": "entidad de la comunidad 1",
    "predicate": "relación inferida",
    "object": "entidad de la comunidad 2"
  }},
  ...
]

Solo incluye relaciones altamente plausibles con predicados claros.
IMPORTANTE: Las relaciones inferidas (predicados) DEBEN tener un máximo de 3 palabras. Preferiblemente 1-2 palabras. Nunca más de 3.
Para predicados, usa frases cortas que describan claramente la relación.
IMPORTANTE: Asegúrate de que el sujeto y objeto sean entidades diferentes - evita auto-referencias.
"""

# Fase 4: Prompts de inferencia de relaciones dentro de comunidades
WITHIN_COMMUNITY_INFERENCE_SYSTEM_PROMPT = """
Eres un experto en representación del conocimiento e inferencia.
Tu tarea es inferir relaciones plausibles entre entidades semánticamente relacionadas que aún no están conectadas en un grafo de conocimiento.
"""

def get_within_community_inference_user_prompt(pairs_text, triples_text):
    return f"""
Tengo un grafo de conocimiento con varias entidades que parecen estar semánticamente relacionadas pero no están directamente conectadas.

Aquí hay algunos pares de entidades que podrían estar relacionadas:
{pairs_text}

Aquí hay algunas relaciones existentes que involucran estas entidades:
{triples_text}

Por favor infiere relaciones plausibles entre estos pares desconectados.
Devuelve tu respuesta como un arreglo JSON de tripletes en el siguiente formato:

[
  {{
    "subject": "entidad1",
    "predicate": "relación inferida",
    "object": "entidad2"
  }},
  ...
]

Solo incluye relaciones altamente plausibles con predicados claros.
IMPORTANTE: Las relaciones inferidas (predicados) DEBEN tener un máximo de 3 palabras. Preferiblemente 1-2 palabras. Nunca más de 3.
IMPORTANTE: Asegúrate de que el sujeto y objeto sean entidades diferentes - evita auto-referencias.
""" 