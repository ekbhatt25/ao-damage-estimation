import React, { useRef, useEffect } from 'react';

const SEVERITY_COLORS = {
    minor:    '#EAB308',  // yellow
    moderate: '#F97316',  // orange
    severe:   '#EF4444',  // red
};

const getColor = (severity) =>
    SEVERITY_COLORS[severity?.toLowerCase()] ?? '#3B82F6';

const ImageOverlay = ({ imageUrl, detections = [] }) => {
    const imgRef  = useRef(null);
    const canvasRef = useRef(null);

    const drawBoxes = () => {
        const img    = imgRef.current;
        const canvas = canvasRef.current;
        if (!img || !canvas || !detections.length) return;

        const scaleX = img.clientWidth  / img.naturalWidth;
        const scaleY = img.clientHeight / img.naturalHeight;

        canvas.width  = img.clientWidth;
        canvas.height = img.clientHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        detections.forEach((det) => {
            if (!det.bbox) return;
            const [x1, y1, x2, y2] = det.bbox;
            const sx = x1 * scaleX;
            const sy = y1 * scaleY;
            const sw = (x2 - x1) * scaleX;
            const sh = (y2 - y1) * scaleY;

            const color = getColor(det.severity);

            // Box
            ctx.strokeStyle = color;
            ctx.lineWidth   = 2;
            ctx.strokeRect(sx, sy, sw, sh);

            // Label background
            const label = `${det.part} — ${det.damage_type}`;
            ctx.font = 'bold 11px sans-serif';
            const textW = ctx.measureText(label).width;
            const padX = 4, padY = 3, textH = 11;
            const labelY = sy > textH + padY * 2 + 2 ? sy - textH - padY * 2 - 2 : sy + 2;

            ctx.fillStyle = color;
            ctx.fillRect(sx, labelY, textW + padX * 2, textH + padY * 2);

            // Label text
            ctx.fillStyle = '#000000';
            ctx.fillText(label, sx + padX, labelY + textH + padY - 2);
        });
    };

    useEffect(() => {
        const img = imgRef.current;
        if (!img) return;
        if (img.complete) {
            drawBoxes();
        } else {
            img.addEventListener('load', drawBoxes);
            return () => img.removeEventListener('load', drawBoxes);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [imageUrl, detections]);

    if (!imageUrl) return null;

    return (
        <div className="relative w-full rounded-xl overflow-hidden border border-gray-700">
            <img
                ref={imgRef}
                src={imageUrl}
                alt="Uploaded vehicle"
                className="w-full block"
                onLoad={drawBoxes}
            />
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
            />
        </div>
    );
};

export default ImageOverlay;
