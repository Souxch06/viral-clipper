import { useState, useEffect } from 'react';
import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then(res => res.json());

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export default function Home() {
  const [url, setUrl] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    // Autofill from share-target redirect or ?shared= param
    try {
      const params = new URLSearchParams(typeof window !== 'undefined' ? window.location.search : '');
      const shared = params.get('shared') || params.get('url') || params.get('q');
      if (shared) setUrl(shared);
    } catch (e) {
      // noop
    }
  }, []);

  const { data: jobStatus } = useSWR(
    jobId ? `${API_BASE || ''}/api/jobs/${jobId}/status` : null,
    fetcher,
    { refreshInterval: 2000 }
  );

  async function submit() {
    if (!url) return alert('Colle une URL YouTube.');
    setBusy(true);
    try {
      const endpoint = `${API_BASE || ''}/api/jobs/analyze`;
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      const json = await res.json();
      setJobId(String(json.job_id));
    } catch (e) {
      console.error(e);
      alert('Erreur lors de la création du job.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ padding: 20, fontFamily: 'Inter,system-ui', background: '#071024', color:'#fff', minHeight:'100vh' }}>
      <h1 style={{ fontSize: 22, marginBottom: 8 }}>AI CLIPPER</h1>
      <p style={{ opacity: 0.9 }}>Transforme tes vidéos en Shorts — optimisé Android (PWA + partage)</p>

      <div style={{ marginTop: 16 }}>
        <label style={{ display: 'block', marginBottom: 8 }}>🔗 Colle ton lien YouTube</label>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://www.youtube.com/watch?v=XXXX"
          style={{
            width: '100%', padding: 12, borderRadius: 10, border: '1px solid #233044',
            background:'#0b1220', color:'#fff', fontSize:16
          }}
        />
      </div>

      <div style={{ marginTop: 14, display:'flex', gap:8 }}>
        <button onClick={submit} disabled={busy} style={{
          flex: 1, padding: 14, borderRadius: 10, background: '#ff6b6b', color: '#000', fontWeight:700
        }}>
          {busy ? 'En cours…' : 'Auto Viral'}
        </button>
        <button onClick={() => alert('Upload non implémenté (backend attendu)')} style={{
          padding: 14, borderRadius: 10, background: '#0b1727', border:'1px solid #233044', color:'#fff'
        }}>Importer</button>
      </div>

      <section style={{ marginTop: 20 }}>
        <h2 style={{ fontSize:18 }}>Statut du job</h2>
        {!jobId && <p style={{ opacity: 0.8 }}>Aucun job en cours. Partage une vidéo depuis YouTube (Chrome → Partager → ViralClipper) ou colle l'URL.</p>}
        {jobId && (
          <div style={{ background:'#071a2a', padding:12, borderRadius:8 }}>
            <div>Job ID: {jobId}</div>
            <div>Status: {jobStatus?.status ?? 'en attente'}</div>
            <div>Étape: {jobStatus?.step ?? '-'}</div>
            <div style={{ marginTop:8 }}>
              {jobStatus?.metadata?.clips ? (
                <div>
                  <h3>Clips générés</h3>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
                    {jobStatus.metadata.clips.map((c: any) => (
                      <div key={c.id} style={{ background:'#021627', padding:8, borderRadius:6 }}>
                        <div style={{ fontSize:12, opacity:0.8 }}>⭐ {c.score}</div>
                        <video src={c.path} controls style={{ width:'100%', borderRadius:6 }} />
                        <a href={c.path} style={{ display:'block', marginTop:6, color:'#ffd166' }} download>Télécharger</a>
                      </div>
                    ))}
                  </div>
                </div>
              ) : <div>Traitement…</div>}
            </div>
          </div>
        )}
      </section>

      <footer style={{ marginTop:24, opacity:0.7, fontSize:12 }}>
        Conseils Android: Installe l’app via "Ajouter à l’écran d’accueil" pour un accès rapide et partage depuis l'app YouTube.
      </footer>
    </main>
  );
}
