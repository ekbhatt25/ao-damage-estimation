import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertCircle, RefreshCw, ShieldCheck, ShieldAlert, TrendingUp, AlertTriangle, Pencil, Check, X, RotateCcw, Info, History, Download, Trash2 } from 'lucide-react';
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
const STATE_OPTIONS = [
    ["AL","Alabama"],["AK","Alaska"],["AZ","Arizona"],["AR","Arkansas"],["CA","California"],
    ["CO","Colorado"],["CT","Connecticut"],["DE","Delaware"],["FL","Florida"],["GA","Georgia"],
    ["HI","Hawaii"],["ID","Idaho"],["IL","Illinois"],["IN","Indiana"],["IA","Iowa"],
    ["KS","Kansas"],["KY","Kentucky"],["LA","Louisiana"],["ME","Maine"],["MD","Maryland"],
    ["MA","Massachusetts"],["MI","Michigan"],["MN","Minnesota"],["MS","Mississippi"],["MO","Missouri"],
    ["MT","Montana"],["NE","Nebraska"],["NV","Nevada"],["NH","New Hampshire"],["NJ","New Jersey"],
    ["NM","New Mexico"],["NY","New York"],["NC","North Carolina"],["ND","North Dakota"],["OH","Ohio"],
    ["OK","Oklahoma"],["OR","Oregon"],["PA","Pennsylvania"],["RI","Rhode Island"],["SC","South Carolina"],
    ["SD","South Dakota"],["TN","Tennessee"],["TX","Texas"],["UT","Utah"],["VT","Vermont"],
    ["VA","Virginia"],["WA","Washington"],["WV","West Virginia"],["WI","Wisconsin"],["WY","Wyoming"],
];
const DAMAGE_OPTIONS = ["Dent","Scratch","Crack","Glass Shatter","Lamp Broken","Tire Flat"];
const SEV_OPTIONS    = ["minor","moderate","severe"];
const SEV_LABELS     = { minor: "Minor", moderate: "Moderate", severe: "Severe" };

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
const ResultsDisplay = ({ results, imageUrl, onReset, sessionId = '' }) => {
    const [editingIdx,      setEditingIdx]      = useState(null);
    const [pendingEdit,     setPendingEdit]     = useState({});
    const [overrides,       setOverrides]       = useState({});   // idx → { part, damage_type, severity, cost_range, action }
    const [showClaimRecord, setShowClaimRecord] = useState(false);
    const [showHistory,     setShowHistory]     = useState(false);
    const [claimHistory,    setClaimHistory]    = useState([]);
    const [historyLoading,  setHistoryLoading]  = useState(false);
    const [selectedState,   setSelectedState]   = useState(results?.state || '');
    const [stateCosts,      setStateCosts]      = useState({});   // idx → { cost_range, action }
    const [stateLoading,    setStateLoading]    = useState(false);
    const [removedIdxs,     setRemovedIdxs]     = useState(new Set());
    const [addedParts,      setAddedParts]      = useState([]);   // [{part, damage_type, severity, cost_range, action}]
    const [editingAddedIdx, setEditingAddedIdx] = useState(null);
    const [pendingAddedEdit,setPendingAddedEdit]= useState({});

    const refetchAllCosts = async (newState) => {
        if (!cost?.damaged_parts?.length) return;
        setStateLoading(true);
        const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
        const newStateCosts = {};
        await Promise.all(cost.damaged_parts.map(async (p, i) => {
            const part        = overrides[i]?.part        ?? p.part;
            const damage_type = overrides[i]?.damage_type ?? p.damage_type;
            const severity    = overrides[i]?.severity    ?? p.severity ?? 'moderate';
            try {
                const res = await fetch(
                    `${API_URL}/estimate?part=${encodeURIComponent(part)}&damage_type=${encodeURIComponent(damage_type)}&severity=${encodeURIComponent(severity)}&state=${newState}`
                );
                const data = await res.json();
                newStateCosts[i] = { cost_range: data.cost_range, action: data.action };
            } catch { /* keep original on failure */ }
        }));
        setStateCosts(newStateCosts);
        setStateLoading(false);
    };

    const handleStateChange = (newState) => {
        setSelectedState(newState);
        if (newState) refetchAllCosts(newState);
        else setStateCosts({});
    };

    const openHistory = async () => {
        setShowHistory(true);
        setHistoryLoading(true);
        try {
            const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
            const res = await fetch(`${API_URL}/claims?session_id=${sessionId}`);
            const data = await res.json();
            setClaimHistory(data.claims || []);
        } catch {
            setClaimHistory([]);
        } finally {
            setHistoryLoading(false);
        }
    };

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

    // Effective cost range: adjuster override > state-adjusted > original AI; null if removed
    const effectiveCost = (i) => {
        if (removedIdxs.has(i)) return null;
        if (overrides[i])  return overrides[i].cost_range;
        if (stateCosts[i]) return stateCosts[i].cost_range;
        return cost?.damaged_parts?.[i]?.cost_range ?? null;
    };

    const effectiveAction = (i) => {
        if (overrides[i])  return overrides[i].action;
        if (stateCosts[i]) return stateCosts[i].action;
        return cost?.damaged_parts?.[i]?.action ?? null;
    };

    // Recalculate total from all effective per-part costs + adjuster-added parts
    const overriddenTotal = (() => {
        if (!cost?.damaged_parts?.length && !addedParts.length) return null;
        let lo = 0, hi = 0;
        cost.damaged_parts?.forEach((_, i) => {
            const r = effectiveCost(i);
            if (r) { lo += r[0]; hi += r[1]; }
        });
        addedParts.forEach(p => {
            if (p.cost_range) { lo += p.cost_range[0]; hi += p.cost_range[1]; }
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

    const applyEdit = async (i) => {
        const { part, damage_type, severity } = pendingEdit;
        try {
            const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
            const st = selectedState || results?.state || '';
            const res = await fetch(
                `${API_URL}/estimate?part=${encodeURIComponent(part)}&damage_type=${encodeURIComponent(damage_type)}&severity=${encodeURIComponent(severity)}&state=${st}`
            );
            const data = await res.json();
            setOverrides(prev => ({ ...prev, [i]: { part, damage_type, severity, cost_range: data.cost_range, action: data.action } }));
        } catch {
            const { cost_range, action } = calcCost(part, damage_type, severity);
            setOverrides(prev => ({ ...prev, [i]: { part, damage_type, severity, cost_range, action } }));
        }
        setEditingIdx(null);
    };

    const cancelEdit = () => setEditingIdx(null);

    const startAddPart = () => {
        const defaults = { part: "Front-bumper", damage_type: "Dent", severity: "moderate" };
        setAddedParts(prev => [...prev, { ...defaults, cost_range: null, action: null }]);
        setEditingAddedIdx(addedParts.length);
        setPendingAddedEdit(defaults);
    };

    const applyAddedEdit = async (idx) => {
        const { part, damage_type, severity } = pendingAddedEdit;
        try {
            const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
            const st = selectedState || results?.state || '';
            const res = await fetch(
                `${API_URL}/estimate?part=${encodeURIComponent(part)}&damage_type=${encodeURIComponent(damage_type)}&severity=${encodeURIComponent(severity)}&state=${st}`
            );
            const data = await res.json();
            setAddedParts(prev => prev.map((p, i) => i === idx ? { part, damage_type, severity, cost_range: data.cost_range, action: data.action } : p));
        } catch {
            const { cost_range, action } = calcCost(part, damage_type, severity);
            setAddedParts(prev => prev.map((p, i) => i === idx ? { part, damage_type, severity, cost_range, action } : p));
        }
        setEditingAddedIdx(null);
    };

    const removeAddedPart = (idx) => setAddedParts(prev => prev.filter((_, i) => i !== idx));

    const resetOverride = (i) => {
        setOverrides(prev => { const next = { ...prev }; delete next[i]; return next; });
        setEditingIdx(null);
    };

    return (<>
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-4xl mx-auto bg-gray-800/50 backdrop-blur-md border border-gray-700 rounded-2xl overflow-hidden shadow-2xl"
        >
            {/* Header */}
            <div className="p-8 border-b border-gray-700 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                        {error ? <AlertCircle className="w-8 h-8 text-red-400" /> : <CheckCircle className="w-8 h-8 text-green-400" />}
                        <h2 className={`text-2xl font-bold ${error ? 'text-red-400' : 'text-white'}`}>
                            {error ? 'Analysis Error' : 'Analysis Complete'}
                        </h2>
                    </div>
                    <div className="flex items-center gap-4">
                        {claim_id && (
                            <div className="relative">
                                <button onClick={() => setShowClaimRecord(v => !v)}
                                    className="text-gray-400 hover:text-white transition-colors"
                                    title="Claim record">
                                    <Info className="w-5 h-5" />
                                </button>
                                {showClaimRecord && (
                                    <div className="absolute right-0 top-7 z-20 w-64 p-3 bg-gray-900 border border-gray-700 rounded-xl shadow-xl text-xs text-gray-400 space-y-1">
                                        <p className="font-medium text-gray-300">Claim Record</p>
                                        <p>ID: <span className="text-gray-200 font-mono break-all">{claim_id}</span></p>
                                        {model_version && <p>Model: <span className="text-gray-200">v{model_version}</span></p>}
                                        {inference_ms != null && <p>Analyzed in: <span className="text-gray-200">{inference_ms}ms</span></p>}
                                    </div>
                                )}
                            </div>
                        )}
                        <button onClick={onReset} className="text-gray-400 hover:text-white transition-colors" title="Close">
                            <X className="w-6 h-6" />
                        </button>
                    </div>
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
                    <ImageOverlay imageUrl={imageUrl} detections={detections
                        .map((det, i) => ({
                            ...det,
                            _idx: i,
                            severity:    overrides[i]?.severity    ?? det.severity,
                            part:        overrides[i]?.part        ?? det.part,
                            damage_type: overrides[i]?.damage_type ?? det.damage_type,
                        }))
                        .filter((_, i) => !removedIdxs.has(i))
                    } />
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

                {/* State selector — re-prices all parts by regional labor rates */}
                {cost?.damaged_parts?.length > 0 && (
                    <div className="flex items-center gap-3 p-3 bg-gray-900/50 rounded-xl border border-gray-700">
                        <label className="text-xs font-medium text-gray-400 whitespace-nowrap">Estimate by state</label>
                        <select
                            value={selectedState}
                            onChange={e => handleStateChange(e.target.value)}
                            className="flex-1 bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1"
                        >
                            <option value="">— national average —</option>
                            {STATE_OPTIONS.map(([abbr, name]) => (
                                <option key={abbr} value={abbr}>{name} ({abbr})</option>
                            ))}
                        </select>
                        {stateLoading && <span className="text-xs text-gray-400 animate-pulse whitespace-nowrap">Updating…</span>}
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
                                {(Object.keys(overrides).length > 0 || Object.keys(stateCosts).length > 0) && (
                                    <div className="flex items-center justify-between mt-1">
                                        <p className="text-orange-400 text-xs">
                                            {Object.keys(overrides).length > 0 ? 'Adjuster adjusted' : `Rates: ${selectedState}`}
                                        </p>
                                        <button
                                            onClick={() => { setOverrides({}); setStateCosts({}); setRemovedIdxs(new Set()); setAddedParts([]); }}
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
                        <label className="text-xs font-medium text-gray-400 block mb-2">Our Assessment</label>
                        <p className="text-gray-300 text-sm leading-relaxed">{explanation}</p>
                    </div>
                )}

                {detections.length === 0 && !error && (
                    <p className="text-black text-center py-4">No damage detected in this image.</p>
                )}

                {/* Detection cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {detections.map((det, i) => {
                        const isEditing  = editingIdx === i;
                        const isOverride = !!overrides[i];
                        const dispPart   = isEditing ? pendingEdit.part        : (overrides[i]?.part        ?? det.part);
                        const dispDmg    = isEditing ? pendingEdit.damage_type : (overrides[i]?.damage_type ?? det.damage_type);
                        const dispSev    = isEditing ? pendingEdit.severity    : (overrides[i]?.severity    ?? det.severity ?? "moderate");
                        const partCost   = effectiveCost(i);
                        const partAction = effectiveAction(i);
                        const isLastOdd  = detections.length % 2 !== 0 && i === detections.length - 1;

                        const isRemoved = removedIdxs.has(i);

                        return (
                            <div key={i} className={`p-4 bg-gray-900/50 rounded-xl border space-y-3 ${isRemoved ? 'border-red-900/40 opacity-50' : isOverride ? 'border-orange-700/60' : 'border-gray-700'} ${isLastOdd ? 'md:col-span-2 md:w-[calc(50%-0.5rem)] md:mx-auto w-full' : ''}`}>

                            {/* Card header */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-white font-medium text-sm">Detection {i + 1}</span>
                                    {isRemoved && <span className="text-xs text-red-400 bg-red-900/40 px-2 py-0.5 rounded-full">Removed by adjuster</span>}
                                    {!isRemoved && isOverride && <span className="text-xs text-orange-400 bg-orange-900/40 px-2 py-0.5 rounded-full">Adjuster adjusted</span>}
                                </div>
                                {isRemoved
                                    ? <button onClick={() => setRemovedIdxs(prev => { const n = new Set(prev); n.delete(i); return n; })} className="text-gray-400 hover:text-white transition-colors" title="Restore detection">
                                        <RotateCcw className="w-4 h-4" />
                                      </button>
                                    : !isEditing
                                    ? <div className="flex gap-2">
                                        <button onClick={() => startEdit(i, det)} className="text-gray-400 hover:text-white transition-colors" title="Edit detection">
                                            <Pencil className="w-4 h-4" />
                                        </button>
                                        <button onClick={() => setRemovedIdxs(prev => new Set([...prev, i]))} className="text-gray-400 hover:text-red-400 transition-colors" title="Remove detection">
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                      </div>
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

                            {/* Part + Damage type — hidden when removed */}
                            {!isRemoved && <><div className="grid grid-cols-2 gap-4">
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
                                            {SEV_OPTIONS.map(s => <option key={s} value={s}>{SEV_LABELS[s]}</option>)}
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
                            </>}
                        </div>
                    );
                })}
                </div>

                {/* Adjuster-added parts */}
                {addedParts.map((added, idx) => {
                    const isEditingAdded = editingAddedIdx === idx;
                    return (
                        <div key={`added-${idx}`} className="p-4 bg-gray-900/50 rounded-xl border border-blue-700/60 space-y-3">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-white font-medium text-sm">Added Part {idx + 1}</span>
                                    <span className="text-xs text-blue-400 bg-blue-900/40 px-2 py-0.5 rounded-full">Added by adjuster</span>
                                </div>
                                <div className="flex gap-2">
                                    {isEditingAdded ? (
                                        <>
                                            <button onClick={() => applyAddedEdit(idx)} className="text-green-400 hover:text-green-300 transition-colors" title="Apply"><Check className="w-4 h-4" /></button>
                                            <button onClick={() => setEditingAddedIdx(null)} className="text-red-400 hover:text-red-300 transition-colors" title="Cancel"><X className="w-4 h-4" /></button>
                                        </>
                                    ) : (
                                        <>
                                            <button onClick={() => { setEditingAddedIdx(idx); setPendingAddedEdit({ part: added.part, damage_type: added.damage_type, severity: added.severity }); }} className="text-gray-400 hover:text-white transition-colors" title="Edit"><Pencil className="w-4 h-4" /></button>
                                            <button onClick={() => removeAddedPart(idx)} className="text-gray-400 hover:text-red-400 transition-colors" title="Remove"><Trash2 className="w-4 h-4" /></button>
                                        </>
                                    )}
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Part</label>
                                    {isEditingAdded
                                        ? <select value={pendingAddedEdit.part} onChange={e => setPendingAddedEdit(p => ({ ...p, part: e.target.value }))} className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1">
                                            {PART_OPTIONS.map(p => <option key={p}>{p}</option>)}
                                          </select>
                                        : <p className="text-white font-medium">{added.part}</p>}
                                </div>
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Damage Type</label>
                                    {isEditingAdded
                                        ? <select value={pendingAddedEdit.damage_type} onChange={e => setPendingAddedEdit(p => ({ ...p, damage_type: e.target.value }))} className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1">
                                            {DAMAGE_OPTIONS.map(d => <option key={d}>{d}</option>)}
                                          </select>
                                        : <p className="text-white font-medium">{added.damage_type}</p>}
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-1">
                                    <label className="text-xs font-medium text-gray-400">Severity</label>
                                    {isEditingAdded
                                        ? <select value={pendingAddedEdit.severity} onChange={e => setPendingAddedEdit(p => ({ ...p, severity: e.target.value }))} className="w-full bg-gray-800 border border-gray-600 text-white text-sm rounded-lg px-2 py-1">
                                            {SEV_OPTIONS.map(s => <option key={s} value={s}>{SEV_LABELS[s]}</option>)}
                                          </select>
                                        : <div className="flex items-center gap-2"><span className="text-white capitalize">{added.severity}</span><div className="flex gap-1">{severityDots(added.severity)}</div></div>}
                                </div>
                                {added.cost_range && (
                                    <div className="space-y-1">
                                        <label className="text-xs font-medium text-gray-400">Part Cost Range</label>
                                        <p className="text-white text-sm">${fmt(added.cost_range[0])} – ${fmt(added.cost_range[1])}{added.action && <span className="text-gray-400 ml-2">({added.action})</span>}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}

                {/* Add Part button */}
                {!error && (
                    <button onClick={startAddPart} className="w-full py-2 border border-dashed border-gray-600 hover:border-blue-500 text-gray-400 hover:text-blue-400 rounded-xl text-sm transition-colors flex items-center justify-center gap-2">
                        + Add Part
                    </button>
                )}

                <div className="pt-2 border-t border-gray-700 space-y-3">
                    <button onClick={onReset}
                        className="w-full py-4 bg-white text-gray-900 rounded-xl font-bold hover:bg-gray-200 transition-colors flex items-center justify-center gap-2">
                        {error ? <X className="w-5 h-5" /> : <RefreshCw className="w-5 h-5" />}
                        {error ? "Dismiss & Go Back" : "Analyze Another Image"}
                    </button>
                    {!error && (
                        <button onClick={openHistory}
                            className="w-full py-4 bg-white text-gray-900 rounded-xl font-bold hover:bg-gray-200 transition-colors flex items-center justify-center gap-2">
                            <History className="w-5 h-5" />
                            View Claim History
                        </button>
                    )}
                </div>
            </div>
        </motion.div>

        {/* Claim History Drawer */}
        {showHistory && (
            <div className="fixed inset-0 z-50 flex justify-end">
                {/* Backdrop */}
                <div className="absolute inset-0 bg-black/60" onClick={() => setShowHistory(false)} />
                {/* Panel */}
                <div className="relative w-full max-w-md bg-gray-900 border-l border-gray-700 h-full overflow-y-auto shadow-2xl flex flex-col">
                    <div className="flex items-center justify-between p-6 border-b border-gray-700">
                        <div className="flex items-center gap-2">
                            <History className="w-5 h-5 text-gray-400" />
                            <h3 className="text-white font-bold text-lg">Claim History</h3>
                        </div>
                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => {
                                    const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
                                    const header = [
                                        'Submission Timestamp',
                                        'Claim ID',
                                        'STP Decision',
                                        'Adjuster Review Required',
                                        'Damaged Parts (Part — Damage Type — Severity)',
                                        'Estimated Repair Cost Low ($)',
                                        'Estimated Repair Cost High ($)',
                                        'AI Confidence (%)',
                                        'Pipeline Version',
                                    ].join(',');
                                    const rows = claimHistory.map(c => {
                                        const parts = (c.damaged_parts || [])
                                            .map(p => `${p.part} — ${p.damage_type} — ${p.severity}`)
                                            .join('; ');
                                        return [
                                            esc(new Date(c.timestamp).toLocaleString()),
                                            esc(c.claim_id),
                                            esc(c.total_loss ? 'Total Loss' : c.stp_eligible ? 'Auto-Approved' : 'Adjuster Review Required'),
                                            esc(c.requires_adjuster_review ? 'Yes' : 'No'),
                                            esc(parts),
                                            c.total_cost_range?.[0] ?? '',
                                            c.total_cost_range?.[1] ?? '',
                                            c.confidence_score != null ? (c.confidence_score * 100).toFixed(0) : '',
                                            esc(c.model_version ? `v${c.model_version}` : ''),
                                        ].join(',');
                                    });
                                    const csv = [header, ...rows].join('\n');
                                    const a = document.createElement('a');
                                    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
                                    a.download = 'claim_history.csv';
                                    a.click();
                                }}
                                className="text-gray-400 hover:text-white transition-colors"
                                title="Export as CSV"
                            >
                                <Download className="w-4 h-4" />
                            </button>
                            <button onClick={() => setShowHistory(false)} className="text-gray-400 hover:text-white transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 p-4 space-y-3">
                        {historyLoading && <p className="text-gray-400 text-sm text-center py-8">Loading...</p>}
                        {!historyLoading && claimHistory.length === 0 && (
                            <p className="text-gray-400 text-sm text-center py-8">No claims on record yet.</p>
                        )}
                        {!historyLoading && claimHistory.map((c) => (
                            <div key={c.claim_id} className="p-4 bg-gray-800 rounded-xl border border-gray-700 text-sm space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                                        c.total_loss ? 'bg-red-900/60 text-red-400'
                                        : c.stp_eligible ? 'bg-green-900/60 text-green-400'
                                        : 'bg-yellow-900/60 text-yellow-400'
                                    }`}>
                                        {c.total_loss ? 'Total Loss' : c.stp_eligible ? 'Auto-Approved' : 'Adjuster Review'}
                                    </span>
                                    <span className="text-gray-500 text-xs">
                                        {new Date(c.timestamp).toLocaleString()}
                                    </span>
                                </div>
                                <p className="text-gray-400 font-mono text-xs truncate">{c.claim_id}</p>
                                {c.total_cost_range?.length === 2 && (
                                    <p className="text-white font-medium">
                                        ${fmt(c.total_cost_range[0])} – ${fmt(c.total_cost_range[1])}
                                    </p>
                                )}
                                {c.confidence_score != null && (
                                    <p className="text-gray-400 text-xs">Confidence: {(c.confidence_score * 100).toFixed(0)}%</p>
                                )}
                            </div>
                        ))}
                    </div>
                    {!historyLoading && claimHistory.length > 0 && (
                        <div className="p-4 border-t border-gray-700">
                            <button
                                onClick={() => setClaimHistory([])}
                                className="w-full py-2 flex items-center justify-center gap-2 text-sm text-gray-400 hover:text-white border border-gray-700 hover:border-gray-500 rounded-xl transition-colors"
                            >
                                Clear view
                            </button>
                        </div>
                    )}
                </div>
            </div>
        )}
    </>);
};

export default ResultsDisplay;
