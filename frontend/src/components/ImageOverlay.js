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

        const dpr    = window.devicePixelRatio || 1;
        const cssW   = img.clientWidth;
        const cssH   = img.clientHeight;

        canvas.width  = cssW * dpr;
        canvas.height = cssH * dpr;
        canvas.style.width  = `${cssW}px`;
        canvas.style.height = `${cssH}px`;

        const scaleX = cssW / img.naturalWidth;
        const scaleY = cssH / img.naturalHeight;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, cssW, cssH);

        // Sort ascending by confidence so highest-confidence box is drawn last (on top)
        const sorted = [...detections]
            .filter(d => d.bbox)
            .sort((a, b) => (a.confidence ?? 0) - (b.confidence ?? 0));

        const scaled = sorted.map((det) => {
            const [x1, y1, x2, y2] = det.bbox;
            return {
                det,
                sx: x1 * scaleX,
                sy: y1 * scaleY,
                sw: (x2 - x1) * scaleX,
                sh: (y2 - y1) * scaleY,
                color: getColor(det.severity),
            };
        });

        // Pass 1: draw all box fills + strokes
        scaled.forEach(({ sx, sy, sw, sh, color }) => {
            ctx.fillStyle = color + '22';  // ~13% opacity fill
            ctx.fillRect(sx, sy, sw, sh);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(sx, sy, sw, sh);
        });

        // Pass 2: draw labels for top 3 by confidence only — avoids clutter with many detections
        const topThree = [...scaled].sort((a, b) => (b.det.confidence ?? 0) - (a.det.confidence ?? 0)).slice(0, 3);
        ctx.font = 'bold 11px sans-serif';
        const padX = 4, padY = 3, textH = 11;
        topThree.forEach(({ det, sx, sy, color }) => {
            const label = `${det.part} — ${det.damage_type}`;
            const textW = ctx.measureText(label).width;
            const bgW = textW + padX * 2;
            const bgH = textH + padY * 2;

            // Place label above box if space, otherwise inside top of box
            const labelY = sy > bgH + 2 ? sy - bgH - 2 : sy + 2;
            // Clamp horizontally so label doesn't run off right edge
            const labelX = Math.min(sx, cssW - bgW - 2);

            ctx.fillStyle = color;
            ctx.fillRect(labelX, labelY, bgW, bgH);
            ctx.fillStyle = '#000000';
            ctx.fillText(label, labelX + padX, labelY + textH + padY - 2);
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
