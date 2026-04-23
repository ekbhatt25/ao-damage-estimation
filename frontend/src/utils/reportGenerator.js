import jsPDF from 'jspdf';

const SEVERITY_COLORS = {
    minor:    { hex: '#EAB308', rgb: [234, 179, 8] },
    moderate: { hex: '#F97316', rgb: [249, 115, 22] },
    severe:   { hex: '#EF4444', rgb: [239, 68, 68] },
};

const AO_BLUE = [8, 68, 119];

function getColor(severity) {
    return SEVERITY_COLORS[severity?.toLowerCase()] ?? SEVERITY_COLORS.moderate;
}

async function renderAnnotatedImage(imageUrl, detections) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const maxW = 1000;
            const scale = img.naturalWidth > maxW ? maxW / img.naturalWidth : 1;
            const canvas = document.createElement('canvas');
            canvas.width  = img.naturalWidth  * scale;
            canvas.height = img.naturalHeight * scale;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            const sorted = [...(detections || [])]
                .filter(d => d.bbox)
                .sort((a, b) => (a.confidence ?? 0) - (b.confidence ?? 0));

            sorted.forEach(det => {
                const [x1, y1, x2, y2] = det.bbox;
                const color = getColor(det.severity);
                const sx = x1 * scale, sy = y1 * scale;
                const sw = (x2 - x1) * scale, sh = (y2 - y1) * scale;
                ctx.fillStyle = color.hex + '33';
                ctx.fillRect(sx, sy, sw, sh);
                ctx.strokeStyle = color.hex;
                ctx.lineWidth = 2;
                ctx.strokeRect(sx, sy, sw, sh);
            });

            const top3 = [...sorted]
                .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))
                .slice(0, 3);

            ctx.font = 'bold 13px sans-serif';
            top3.forEach(det => {
                const [x1, y1] = det.bbox;
                const color = getColor(det.severity);
                const sx = x1 * scale, sy = y1 * scale;
                const label = `${det.part} — ${det.damage_type}`;
                const textW = ctx.measureText(label).width;
                const padX = 4, padY = 3, textH = 13;
                const bgW = textW + padX * 2, bgH = textH + padY * 2;
                const labelY = sy > bgH + 2 ? sy - bgH - 2 : sy + 2;
                const labelX = Math.min(sx, canvas.width - bgW - 2);
                ctx.fillStyle = color.hex;
                ctx.fillRect(labelX, labelY, bgW, bgH);
                ctx.fillStyle = '#000000';
                ctx.fillText(label, labelX + padX, labelY + textH + padY - 2);
            });

            resolve(canvas.toDataURL('image/jpeg', 0.85));
        };
        img.onerror = () => resolve(null);
        img.src = imageUrl;
    });
}

function addPage(doc, result, imgDataUrl, pageNum, totalPages) {
    const margin = 15;
    const pageW  = doc.internal.pageSize.getWidth();
    let y = 0;

    // Blue header bar
    doc.setFillColor(...AO_BLUE);
    doc.rect(0, 0, pageW, 12, 'F');
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text('Auto-Owners Insurance  —  Vehicle Damage Assessment Report', margin, 8);

    y = 20;

    // Title + image label
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(13);
    doc.text('Damage Assessment', margin, y);
    if (totalPages > 1) {
        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(100, 100, 100);
        doc.text(`Image ${pageNum} of ${totalPages}`, pageW - margin - 28, y);
    }

    y += 5;
    doc.setFontSize(7.5);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(130, 130, 130);
    const meta = [
        new Date().toLocaleString(),
        result.claim_id   ? `Claim: ${result.claim_id}`  : null,
        result.state      ? `State: ${result.state}`     : null,
        result.inference_ms != null ? `Analyzed in ${result.inference_ms}ms` : null,
    ].filter(Boolean).join('   ·   ');
    doc.text(meta, margin, y);

    y += 3;
    doc.setDrawColor(210, 210, 210);
    doc.line(margin, y, pageW - margin, y);
    y += 5;

    // Annotated image
    if (imgDataUrl) {
        const props  = doc.getImageProperties(imgDataUrl);
        const maxW   = pageW - margin * 2;
        const maxH   = 72;
        const ratio  = Math.min(maxW / props.width, maxH / props.height);
        const imgW   = props.width  * ratio;
        const imgH   = props.height * ratio;
        doc.addImage(imgDataUrl, 'JPEG', margin, y, imgW, imgH);
        y += imgH + 5;
    }

    // STP banner
    if (result.stp_eligible != null) {
        const stp = result.stp_eligible;
        doc.setFillColor(...(stp ? [220, 252, 231] : [254, 249, 195]));
        doc.setDrawColor(...(stp ? [74, 222, 128] : [250, 204, 21]));
        doc.roundedRect(margin, y, pageW - margin * 2, 9, 1.5, 1.5, 'FD');
        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(...(stp ? [22, 101, 52] : [120, 53, 15]));
        doc.text(stp ? 'Auto-Approved (STP)' : 'Adjuster Review Required', margin + 3, y + 6);
        if (result.stp_reasoning) {
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(7);
            doc.setTextColor(80, 80, 80);
            const short = result.stp_reasoning.length > 90
                ? result.stp_reasoning.slice(0, 90) + '…'
                : result.stp_reasoning;
            doc.text(short, margin + 55, y + 6);
        }
        y += 13;
    }

    // Cost + confidence
    const total = result.cost?.total_cost_range;
    const conf  = result.confidence_score;
    if (total || conf != null) {
        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(0, 0, 0);
        if (total) doc.text(
            `Total Estimated Cost: $${total[0].toLocaleString()} – $${total[1].toLocaleString()}`,
            margin, y
        );
        if (conf != null) doc.text(
            `Confidence: ${(conf * 100).toFixed(0)}%`,
            pageW - margin - 38, y
        );
        y += 7;
    }

    // Detections table
    const parts = result.cost?.damaged_parts || [];
    if (parts.length > 0) {
        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(0, 0, 0);
        doc.text('Detected Damage', margin, y);
        y += 4;

        const cols    = [margin, margin + 38, margin + 72, margin + 97, margin + 118, margin + 148];
        const headers = ['Part', 'Damage Type', 'Severity', 'Action', 'Cost Range', 'Labor Rate'];

        doc.setFillColor(235, 238, 245);
        doc.rect(margin, y, pageW - margin * 2, 6, 'F');
        doc.setFontSize(7.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(40, 40, 40);
        headers.forEach((h, i) => doc.text(h, cols[i] + 1, y + 4.2));
        y += 6;

        doc.setFont('helvetica', 'normal');
        parts.forEach((p, i) => {
            if (y > 268) { doc.addPage(); y = margin; }
            if (i % 2 === 0) {
                doc.setFillColor(248, 249, 251);
                doc.rect(margin, y, pageW - margin * 2, 6, 'F');
            }
            doc.setTextColor(0, 0, 0);
            const costStr = p.cost_range
                ? `$${p.cost_range[0].toLocaleString()} – $${p.cost_range[1].toLocaleString()}`
                : '—';
            const laborStr = p.labor_rate ? `$${p.labor_rate}/hr` : '—';
            [p.part, p.damage_type, p.severity ?? '—', p.action ?? '—', costStr, laborStr]
                .forEach((cell, ci) => doc.text(String(cell), cols[ci] + 1, y + 4.2));
            y += 6;
        });
        y += 5;
    }

    // AI explanation
    if (result.explanation && y < 255) {
        if (y > 250) return;
        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(0, 0, 0);
        doc.text('AI Assessment', margin, y);
        y += 4;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(8);
        doc.setTextColor(60, 60, 60);
        const lines = doc.splitTextToSize(result.explanation, pageW - margin * 2);
        doc.text(lines, margin, y);
        y += lines.length * 4 + 4;
    }

    // Fraud flags
    if (result.fraud_flags?.length > 0 && y < 270) {
        doc.setFontSize(8.5);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(180, 80, 0);
        doc.text('Fraud Signals Detected', margin, y);
        y += 4;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(8);
        doc.setTextColor(100, 50, 0);
        result.fraud_flags.forEach(flag => {
            doc.text(`• ${flag}`, margin + 2, y);
            y += 4;
        });
    }
}

export async function generatePDF(resultsList, imageUrls, indices) {
    const doc = new jsPDF({ unit: 'mm', format: 'a4' });

    for (let i = 0; i < indices.length; i++) {
        if (i > 0) doc.addPage();
        const idx = indices[i];
        const imgDataUrl = await renderAnnotatedImage(
            imageUrls[idx],
            resultsList[idx]?.detections || []
        );
        addPage(doc, resultsList[idx] || {}, imgDataUrl, i + 1, indices.length);
    }

    doc.save(`ao-damage-report-${new Date().toISOString().slice(0, 10)}.pdf`);
}

export function generateCSV(resultsList, imageUrls, indices) {
    const esc = v => `"${String(v ?? '').replace(/"/g, '""')}"`;

    const headers = [
        'Image', 'Claim ID', 'Date', 'State',
        'Part', 'Damage Type', 'Severity', 'Action',
        'Cost Low ($)', 'Cost High ($)', 'Labor Rate ($/hr)',
        'Total Cost Low ($)', 'Total Cost High ($)',
        'Confidence (%)', 'STP Eligible', 'Adjuster Review Required',
        'Fraud Flags',
    ];

    const rows = [];
    indices.forEach((idx, imgNum) => {
        const r     = resultsList[idx] || {};
        const parts = r.cost?.damaged_parts || [];
        const date  = new Date().toLocaleDateString();
        const total = r.cost?.total_cost_range;
        const conf  = r.confidence_score != null ? (r.confidence_score * 100).toFixed(0) : '';
        const fraud = (r.fraud_flags || []).join('; ');

        const sharedCols = (pi) => pi === 0
            ? [imgNum + 1, esc(r.claim_id || ''), esc(date), esc(r.state || '')]
            : ['', '', '', ''];

        const summaryCols = (pi) => pi === 0
            ? [total?.[0] ?? '', total?.[1] ?? '', conf,
               esc(r.stp_eligible ? 'Yes' : 'No'),
               esc(r.requires_adjuster_review ? 'Yes' : 'No'),
               esc(fraud)]
            : ['', '', '', '', '', ''];

        if (parts.length === 0) {
            rows.push([
                ...sharedCols(0),
                '', '', '', '', '', '', '',
                ...summaryCols(0),
            ].join(','));
        } else {
            parts.forEach((p, pi) => {
                rows.push([
                    ...sharedCols(pi),
                    esc(p.part), esc(p.damage_type), esc(p.severity ?? ''), esc(p.action ?? ''),
                    p.cost_range?.[0] ?? '', p.cost_range?.[1] ?? '', p.labor_rate ?? '',
                    ...summaryCols(pi),
                ].join(','));
            });
        }
    });

    const csv  = [headers.join(','), ...rows].join('\n');
    const link = document.createElement('a');
    link.href     = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    link.download = `ao-damage-report-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
}
