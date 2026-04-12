import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertCircle, RefreshCw, ShieldCheck, ShieldAlert, TrendingUp, AlertTriangle, Pencil, Check, X, RotateCcw } from 'lucide-react';
import ImageOverlay from './ImageOverlay';

// ── Client-side cost lookup (mirrors backend cost_estimator.py) ───────────────
const FALLBACK_COSTS = {
    "Front-bumper":    [350,  900],
    "Back-bumper":     [300,  800],
    "Hood":            [600, 1200],
    "Front-door":      [500, 1100],
    "Back-door":       [450, 1000],
    "Fender":          [400,  850],
    "Windshield":      [null, 350],
    "Back-windshield": [null, 300],
    "Front-window":    [null, 200],
    "Back-window":     [null, 200],
    "Headlight":       [null, 275],
    "Tail-light":      [null, 175],
    "Mirror":          [null, 200],
    "Grille":          [150,  250],
    "Roof":            [700, 2000],
    "Trunk":           [400,  900],
    "Quarter-panel":   [600, 1100],
    "Rocker-panel":    [300,  600],
    "Front-wheel":     [null, 250],
    "Back-wheel":      [null, 250],
    "License-plate":   [null,  50],
};
const SEVERITY_MULT  = { minor: 0.65, moderate: 1.00, severe: 1.45 };
const REPLACE_ALWAYS = new Set(["Glass Shatter", "Lamp Broken", "Tire Flat"]);
const REPLACE_ONLY   = new Set(["Windshield", "Back-windshield", "Front-window", "Back-window",
                                 "Headlight", "Tail-light", "Front-wheel", "Back-wheel", "License-plate"]);

const PART_OPTIONS   = ["Back-bumper","Back-door","Back-wheel","Back-window","Back-windshield",
                         "Fender","Front-bumper","Front-door","Front-wheel","Front-window",
                         "Grille","Headlight","Hood","License-plate","Mirror",
                         "Quarter-panel","Rocker-panel","Roof","Tail-light","Trunk","Windshield"];
const DAMAGE_OPTIONS = ["Dent","Scratch","Crack","Glass Shatter","Lamp Broken","Tire Flat"];
const SEV_OPTIONS    = ["minor","moderate","severe"];

function calcCost(part, damageType, severity) {
    const [repairMid, replaceMid] = FALLBACK_COSTS[part] ?? [300, 600];
    const shouldReplace = REPLACE_ALWAYS.has(damageType) || REPLACE_ONLY.has(part) || repairMid === null;
    const base   = shouldReplace ? replaceMid : repairMid;
    const cost   = base * (SEVERITY_MULT[severity] ?? 1.0);
    return {
        cost_range: [Math.round(cost * 0.85), Math.round(cost * 1.15)],
        action:     shouldReplace ? "replace" : "repair",
    };
}

// ── Helpers ────────────────────────────────────────────────────────────────────
const severityDots = (severity) => {
    const level = { minor: 1, moderate: 2, severe: 3 }[severity?.toLowerCase()] ?? 0;
    return [1, 2, 3].map(i => (
        <div key={i} className={`w-2 h-2 rounded-full ${i <= level ? 'bg-yellow-500' : 'bg-gray-700'}`} />
    ));
};

const fmt = (n) => n?.toLocaleString() ?? '—';

// ── Component ──────────────────────────────────────────────────────────────────
const ResultsDisplay = ({ results, imageUrl, onReset }) => {
    const [editingIdx,    setEditingIdx]    = useState(null);
    const [pendingEdit,   setPendingEdit]   = useState({});
    const [overrides,     setOverrides]     = useState({});   // idx → { part, damage_type, severity, cost_range, action }

    if (!results) return null;

    const {
        detections = [],
        summary = {},
        error,
        stp_eligible,
        stp_reasoning,
        confidence_score,
        total_loss,
        explanation,
        cost,
        claim_id,
        fraud_flags = [],
        model_version,
        inference_ms,
    } = results;

    // Effective cost range for a detection (override if present, else original)
    const effectiveCost = (i) => {
        if (overrides[i]) return overrides[i].cost_range;
        return cost?.damaged_parts?.[i]?.cost_range ?? null;
    };

    const effectiveAction = (i) => {
        if (overrides[i]) return overrides[i].action;
        return cost?.damaged_parts?.[i]?.action ?? null;
    };

    // Recalculate total from all effective per-part costs
    const overriddenTotal = (() => {
        if (!cost?.damaged_parts?.length) return null;
        let lo = 0, hi = 0;
        cost.damaged_parts.forEach((_, i) => {
            const r = effectiveCost(i);
            if (r) { lo += r[0]; hi += r[1]; }
        });
        return [lo, hi];
    })();

    const displayTotal = overriddenTotal ?? cost?.total_cost_range;

    const startEdit = (i, det) => {
        setEditingIdx(i);
        setPendingEdit({
            part:        overrides[i]?.part        ?? det.part,
            damage_type: overrides[i]?.damage_type ?? det.damage_type,
            severity:    overrides[i]?.severity    ?? det.severity ?? "moderate",
        });
    };

    const applyEdit = (i) => {
        const { part, damage_type, severity } = pendingEdit;
        const { cost_range, action } = calcCost(part, damage_type, severity);
        setOverrides(prev => ({ ...prev, [i]: { part, damage_type, severity, cost_range, action } }));
        setEditingIdx(null);
    };

    const cancelEdit = () => setEditingIdx(null);

    const resetOverride = (i) => {
        setOverrides(prev => { const next = { ...prev }; delete next[i]; return next; });
        setEditingIdx(null);
    };

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-2xl mx-auto bg-gray-800/50 backdrop-blur-md border border-gray-700 rounded-2xl overflow-hidden shadow-2xl"
        >
            {/* Header */}
            <div className="p-8 border-b border-gray-700 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
                <div className="flex items-center gap-3 mb-2">
                    {error ? <AlertCircle className="w-8 h-8 text-red-400" /> : <CheckCircle className="w-8 h-8 text-green-400" />}
                    <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
                </div>
                {error
                    ? <p className="text-red-400 ml-11">{error}</p>
                    : <p className="text-black ml-11">
                        {summary.total_damaged_parts ?? 0} damaged part{summary.total_damaged_parts !== 1 ? 's' : ''} detected
                      </p>
                }
            </div>

            <div className="p-8 space-y-6">

                {/* Image overlay */}
                {imageUrl && detections.length > 0 && (
                    <ImageOverlay imageUrl={imageUrl} detections={detections} />
                )}

                {/* Fraud Flags */}
                {fraud_flags.length > 0 && (
                    <div className="p-4 rounded-xl border bg-orange-900/30 border-orange-700">
                        <div className="flex items-center gap-2 mb-2">
                            <AlertTriangle className="w-4 h-4 text-orange-400 flex-shrink-0" />
                            <p className="text-orange-400 font-bold text-sm">Image Fraud Signals Detected</p>
                        </div>
                        <ul className="space-y-1">
                            {fraud_flags.map((flag, i) => {
                                const [name, detail] = flag.split(' (');
                                return (
                                    <li key={i} className="text-xs">
                                        <span className="text-orange-300 font-medium">{name.replace(/_/g, ' ')}</span>
                                        {detail && <span className="text-gray-200"> — {detail.replace(')', '')}</span>}
                                    </li>
                                );
                            })}
                        </ul>
                    </div>
                )}

                {/* STP Banner */}
                {stp_eligible != null && (
                    <div className={`p-4 rounded-xl border flex items-start gap-3 ${
                        total_loss ? 'bg-red-900/30 border-red-700'
                        : stp_eligible ? 'bg-green-900/30 border-green-700'
                        : 'bg-yellow-900/30 border-yellow-700'
                    }`}>
                        {total_loss
                            ? <TrendingUp className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                            : stp_eligible
                                ? <ShieldCheck className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                                : <ShieldAlert className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
                        }
                        <div>
                            <p className={`font-bold text-sm ${total_loss ? 'text-red-400' : stp_eligible ? 'text-green-400' : 'text-yellow-400'}`}>
                                {total_loss ? 'Total Loss' : stp_eligible ? 'Auto-Approved' : 'Adjuster Review Required'}
                            </p>
                            {stp_reasoning && <p className="text-gray-300 text-xs mt-1">{stp_reasoning}</p>}
                        </div>
                    </div>
                )}

                {/* Cost + Confidence */}
                {(cost || confidence_score != null) && (
                    <div className="grid grid-cols-2 gap-4">
                        {displayTotal && (
                            <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                                <label className="text-xs font-medium text-gray-400">Estimated Repair Cost</label>
                                <p className="text-white font-bold text-lg mt-1">
                                    ${fmt(displayTotal[0])} – ${fmt(displayTotal[1])}
                                </p>
                                {Object.keys(overrides).length > 0 && (
                                    <div className="flex items-center justify-between mt-1">
                                        <p className="text-orange-400 text-xs">Adjuster adjusted</p>
                                        <button
                                            onClick={() => setOverrides({})}
                                            className="text-xs text-gray-400 hover:text-white flex items-center gap-1 transition-colors"
                                            title="Reset all overrides to original AI estimates"
                                        >
                                            <RotateCcw className="w-3 h-3" />
                                            Reset all
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}
                        {confidence_score != null && (
                            <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                                <label className="text-xs font-medium text-gray-400">Claim Confidence</label>
                                <p className="text-white font-bold text-lg mt-1">
                                    {(confidence_score * 100).toFixed(0)}%
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* Explanation */}
                {explanation && (
                    <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                        <label className="text-xs font-medium text-gray-400 block mb-2">AI Assessment</label>
                        <p className="text-gray-300 text-sm leading-relaxed">{explanation}</p>
                    </div>
                )}

                {detections.length === 0 && !error && (
                    <p className="text-black text-center py-4">No damage detected in this image.</p>
                )}

                {/* Detection cards */}
                {detections.map((det, i) => {
                    const isEditing  = editingIdx === i;
                    const isOverride = !!overrides[i];
                    const dispPart   = isEditing ? pendingEdit.part        : (overrides[i]?.part        ?? det.part);
                    const dispDmg    = isEditing ? pendingEdit.damage_type : (overrides[i]?.damage_type ?? det.damage_type);
                    const dispSev    = isEditing ? pendingEdit.severity    : (overrides[i]?.severity    ?? det.severity ?? "moderate");
                    const partCost   = effectiveCost(i);
                    const partAction = effectiveAction(i);

                    return (
                        <div key={i} className={`p-4 bg-gray-900/50 rounded-xl border space-y-3 ${isOverride ? 'border-orange-700/60' : 'border-gray-700'}`}>

                            {/* Card header */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-white font-medium text-sm">Detection {i + 1}</span>
                                    {isOverride && <span className="text-xs text-orange-400 bg-orange-900/40 px-2 py-0.5 rounded-full">Adjuster adjusted</span>}
                                </div>
                                {!isEditing
                                    ? <button onClick={() => startEdit(i, det)} className="text-gray-400 hover:text-white transition-colors" title="Edit detection">
                                        <Pencil className="w-4 h-4" />
                                      </button>
                                    : <div className="flex gap-2">
                                        <button onClick={() => applyEdit(i)} className="text-green-400 hover:text-green-300 transition-colors" title="Apply">
                                            <Check className="w-4 h-4" />
                                        </button>
                                        {isOverride && (
                                            <button onClick={() => resetOverride(i)} className="text-gray-400 hover:text-white transition-colors" title="Reset to original">
                                                <RotateCcw className="w-4 h-4" />
                                            </button>
                                        )}
                                        <button onClick={cancelEdit} className="text-red-400 hover:text-red-300 transition-colors" title="Cancel">
                                            <X className="w-4 h-4" />
                                        </button>
                                      </div>
                                }
                            </div>

                            {/* Part + Damage type */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Part</label>
                                    {isEditing
                                        ? <select value={pendingEdit.part} onChange={e => setPendingEdit(p => ({ ...p, part: e.target.value }))}
                                            className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1">
                                            {PART_OPTIONS.map(p => <option key={p}>{p}</option>)}
                                          </select>
                                        : <p className="text-white font-medium">{dispPart}</p>
                                    }
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Damage Type</label>
                                    {isEditing
                                        ? <select value={pendingEdit.damage_type} onChange={e => setPendingEdit(p => ({ ...p, damage_type: e.target.value }))}
                                            className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1">
                                            {DAMAGE_OPTIONS.map(d => <option key={d}>{d}</option>)}
                                          </select>
                                        : <p className="text-white font-medium">{dispDmg}</p>
                                    }
                                </div>
                            </div>

                            {/* Severity + Confidence */}
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Severity</label>
                                    {isEditing
                                        ? <select value={pendingEdit.severity} onChange={e => setPendingEdit(p => ({ ...p, severity: e.target.value }))}
                                            className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1">
                                            {SEV_OPTIONS.map(s => <option key={s}>{s}</option>)}
                                          </select>
                                        : <div className="flex items-center gap-2">
                                            <span className="text-white capitalize">{dispSev ?? '—'}</span>
                                            <div className="flex gap-1">{severityDots(dispSev)}</div>
                                          </div>
                                    }
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Confidence</label>
                                    <p className="text-white">{det.confidence != null ? `${(det.confidence * 100).toFixed(0)}%` : '—'}</p>
                                </div>
                            </div>

                            {/* Part cost */}
                            {partCost && (
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Part Cost Range</label>
                                    <p className="text-white text-sm">
                                        ${fmt(partCost[0])} – ${fmt(partCost[1])}
                                        {partAction && <span className="text-gray-400 ml-2">({partAction})</span>}
                                    </p>
                                </div>
                            )}
                        </div>
                    );
                })}

                {claim_id && (
                    <div className="p-3 bg-gray-900/50 rounded-xl border border-gray-700/50 text-xs text-gray-400 space-y-1">
                        <p className="font-medium text-gray-300">Claim Record</p>
                        <p>ID: <span className="text-gray-200 font-mono">{claim_id}</span></p>
                        {model_version && <p>Model: <span className="text-gray-200">v{model_version}</span></p>}
                        {inference_ms != null && <p>Analyzed in: <span className="text-gray-200">{inference_ms}ms</span></p>}
                    </div>
                )}

                <div className="pt-2 border-t border-gray-700">
                    <button onClick={onReset}
                        className="w-full py-4 bg-white text-gray-900 rounded-xl font-bold hover:bg-gray-200 transition-colors flex items-center justify-center gap-2">
                        <RefreshCw className="w-5 h-5" />
                        Analyze Another Image
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

export default ResultsDisplay;
