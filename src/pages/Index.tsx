// import FacialEmotionDetector from '@/components/FacialEmotionDetector';

// const Index = () => {
//   return <FacialEmotionDetector />;
// };

// export default Index;

import FacialEmotionDetector from '@/components/FacialEmotionDetector';

interface IndexProps {
  apiUrl: string;
}

const Index = ({ apiUrl }: IndexProps) => {
  return <FacialEmotionDetector apiUrl={apiUrl} />;
};

export default Index;
