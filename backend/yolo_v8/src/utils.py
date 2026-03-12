def convert_to_json(results, class_names):

    output = {"damaged_parts": []}

    for box in results[0].boxes:

        cls = int(box.cls)
        conf = float(box.conf)

        output["damaged_parts"].append({
            "damage_type": class_names[cls],
            "confidence": conf
        })

    return output