import { Routes, Route } from 'react-router-dom';
import Upload from './pages/Upload';
import Results from './pages/Results';
import Layout from './components/layout/Layout';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Upload />} />
        <Route path="/results/:jobId" element={<Results />} />
      </Routes>
    </Layout>
  );
}

export default App;