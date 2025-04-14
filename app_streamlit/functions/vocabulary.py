def obtener_vocabulario(letra):
    indices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z']

    if letra not in indices:
        return "Esta letra no contiene vocabulario"

    if letra == 'C':
        return ['Carbón: Combustible fósil sólido formado a partir de restos orgánicos que se han descompuesto y comprimido bajo condiciones de alta presión y temperatura a lo largo de millones de años. Se utiliza ampliamente para la generación de energía y como materia prima en la industria, especialmente en la producción de electricidad en centrales térmicas y en la fabricación de acero.',
            'Ciclo combinado: Tecnología de generación de energía eléctrica en la que coexisten dos ciclos termodinámicos en un sistema: uno, cuyo fluido de trabajo es el vapor de agua, y otro, cuyo fluido de trabajo es un gas. En una central eléctrica el ciclo de gas genera energía eléctrica mediante una turbina de gas y el ciclo de vapor de agua lo hace mediante una o varias turbinas de vapor.',
            'Cogeneración: Proceso mediante el cual se obtiene simultáneamente energía eléctrica y energía térmica y/o mecánica útil.']
    elif letra == 'E':
        return ['Eólica: Energía que se genera aprovechando la fuerza del viento, utilizando aerogeneradores que convierten la energía cinética del viento en electricidad.']
    elif letra == 'F':
        return ['Fuel+Gas: Término que suele referirse a sistemas o soluciones que combinan el uso de combustibles tradicionales (como el diésel o el gas natural) y tecnologías de gas.']
    elif letra == 'H':
        return ['Hidráulica: Energía obtenida del aprovechamiento de la fuerza del agua en movimiento, como la de ríos, corrientes o el agua almacenada en embalses.',
            'Hidroeólica: Producción de energía eléctrica a través de la integración de un parque eólico, un grupo de bombeo y una central hidroeléctrica.']
    elif letra == 'M':
        return ['Motores diésel: Motores de combustión interna que funcionan utilizando diésel como combustible.']
    elif letra == 'N':
        return ['Nuclear: Fuente de energía que se genera a partir de las reacciones nucleares, principalmente mediante el proceso de fisión nuclear.']
    elif letra == 'R':
        return ['Residuos no renovables: Son aquellos residuos generados por el uso de recursos naturales que no pueden ser regenerados o renovados en un corto periodo de tiempo.',
            'Residuos renovables: Material orgánico no fósil de origen biológico resultante de los desechos sólidos urbanos y algunos desechos comerciales e industriales no peligrosos.']
    elif letra == 'S':
        return ['Solar fotovoltaica: Luz solar convertida en electricidad mediante el uso de células solares.',
            'Solar térmica: Tecnología que aprovecha la radiación solar para generar calor.']
    elif letra == 'T':
        return ['Turbinación bombeo: Técnica utilizada en plantas hidroeléctricas de bombeo, donde se combina la generación y el almacenamiento de energía.',
            'Turbina de gas: Dispositivo de generación de energía que transforma la energía química de un combustible en energía mecánica y luego en electricidad.',
            'Turbina de Vapor: Dispositivo que convierte la energía térmica del vapor de agua en energía mecánica.']
    else:
        return "Esta letra no contiene vocabulario"
