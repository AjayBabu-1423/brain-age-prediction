

--
-- Database: `brain_age_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL auto_increment,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`id`, `username`, `password`) VALUES
(1, 'admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `doctor`
--

CREATE TABLE `doctor` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(150) NOT NULL,
  `mobile` varchar(20) default NULL,
  `email` varchar(150) default NULL,
  `department` varchar(150) default NULL,
  `location` varchar(150) default NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  `status` enum('Pending','Approved','Rejected') default 'Pending',
  `created_at` timestamp NOT NULL default CURRENT_TIMESTAMP,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `doctor`
--

INSERT INTO `doctor` (`id`, `name`, `mobile`, `email`, `department`, `location`, `username`, `password`, `status`, `created_at`) VALUES
(1, 'Raj', '8124484080', 'raj@gmail.com', 'Neurologist', 'Trichy', 'raj', '1234', 'Approved', '2026-02-23 22:30:20');

-- --------------------------------------------------------

--
-- Table structure for table `patient`
--

CREATE TABLE `patient` (
  `id` int(11) NOT NULL auto_increment,
  `patient_id` varchar(10) NOT NULL,
  `name` varchar(150) NOT NULL,
  `mobile` varchar(20) default NULL,
  `email` varchar(150) default NULL,
  `age` int(11) default NULL,
  `location` varchar(150) default NULL,
  `gender` varchar(20) default NULL,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL default CURRENT_TIMESTAMP,
  PRIMARY KEY  (`id`),
  UNIQUE KEY `patient_id` (`patient_id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `patient`
--

INSERT INTO `patient` (`id`, `patient_id`, `name`, `mobile`, `email`, `age`, `location`, `gender`, `username`, `password`, `created_at`) VALUES
(1, 'P001', 'Raj', '8148956634', 'akil@gmail.com', 22, 'Trichy', 'Male', 'raj', '1234', '2026-02-23 21:41:07');

-- --------------------------------------------------------

--
-- Table structure for table `predictions`
--

CREATE TABLE `predictions` (
  `id` int(11) NOT NULL auto_increment,
  `patient_id` int(11) NOT NULL,
  `actual_age` int(11) NOT NULL,
  `predicted_age` float NOT NULL,
  `brain_age_gap` float NOT NULL,
  `recommendation` text,
  `filename` varchar(255) default NULL,
  `key_aspect` varchar(255) default NULL,
  `causes` text,
  `problems` text,
  `created_at` timestamp NOT NULL default CURRENT_TIMESTAMP,
  `gender` varchar(20) default NULL,
  `memory_loss` varchar(50) default NULL,
  `headache` varchar(50) default NULL,
  `sleep_issues` varchar(50) default NULL,
  `family_history` varchar(50) default NULL,
  `lifestyle` varchar(100) default NULL,
  `risk_level` varchar(50) default NULL,
  `doctor_recommendation` text,
  `xai_explanation` text,
  PRIMARY KEY  (`id`),
  KEY `patient_id` (`patient_id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=32 ;

--
-- Dumping data for table `predictions`
--

INSERT INTO `predictions` (`id`, `patient_id`, `actual_age`, `predicted_age`, `brain_age_gap`, `recommendation`, `filename`, `key_aspect`, `causes`, `problems`, `created_at`, `gender`, `memory_loss`, `headache`, `sleep_issues`, `family_history`, `lifestyle`, `risk_level`, `doctor_recommendation`, `xai_explanation`) VALUES
(12, 1, 40, 40.73, 0.73, 'Low risk: Maintain healthy routine', 'uploads\\20260223224059_OAS1_0011_MR1_mpr_n4_anon_111_t88_gfc.img', 'Normal Ageing', 'Normal ageing', 'None', '2026-02-23 22:40:59', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
(13, 1, 34, 49.66, 15.66, 'High risk: Consult neurologist, exercise, diet, stress management', 'uploads\\20260224093314_brain.img', 'Brain Volume Reduction & Prefrontal Cortex', 'Neuron loss,Chronic stress', 'Frequent forgetfulness,Difficulty in making decisions', '2026-02-24 09:33:15', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
(24, 1, 22, 40.96, 18.96, 'The predicted brain age is significantly higher than the chronological age, suggesting accelerated structural brain ageing. This may be associated with cortical thinning, white matter degradation, and increased risk of cognitive impairment. A neurological evaluation is strongly advised.', 'uploads\\20260224125814_brain.img', 'Structural Changes', 'Accelerated neuronal loss,Chronic stress exposure,Reduced cerebral blood circulation,Neuroinflammatory processes', 'Frequent forgetfulness,Difficulty in planning and decision making,Reduced attention span,Impaired executive functioning', '2026-02-24 12:58:15', NULL, NULL, NULL, NULL, NULL, NULL, 'High Risk', NULL, NULL),
(25, 1, 34, 40.96, 6.96, 'The brain appears slightly older than the actual age, indicating possible early functional changes. While not severe, this may reflect lifestyle factors, mild neurotransmitter imbalance, or early white matter changes. Preventive intervention is recommended.', 'uploads\\20260224152933_brain.img', 'Functional Changes', 'Early white matter degradation,Mild neurotransmitter imbalance,Lifestyle-related risk factors', 'Slower thinking speed,Occasional memory lapses,Reduced concentration', '2026-02-24 15:29:34', NULL, NULL, NULL, NULL, NULL, NULL, 'Moderate Risk', 'Nonetrw25mk3m', NULL),
(26, 1, 35, 49.14, 14.14, 'The predicted brain age is significantly higher than the chronological age, suggesting accelerated structural brain ageing. This may be associated with cortical thinning, white matter degradation, and increased risk of cognitive impairment. A neurological evaluation is strongly advised.', 'uploads\\20260224161824_brain.img', 'Structural Changes', 'Accelerated neuronal loss,Chronic stress exposure,Reduced cerebral blood circulation,Neuroinflammatory processes', 'Frequent forgetfulness,Difficulty in planning and decision making,Reduced attention span,Impaired executive functioning', '2026-02-24 16:18:25', NULL, NULL, NULL, NULL, NULL, NULL, 'High Risk', NULL, 'The model predicts accelerated brain ageing mainly due to abnormal white matter regions that reduce the efficiency of neural signal transmission. This structural degradation is associated with slow cognitive processing and poor motor coordination.');

--
-- Constraints for dumped tables
--

--
-- Constraints for table `predictions`
--
ALTER TABLE `predictions`
  ADD CONSTRAINT `predictions_ibfk_1` FOREIGN KEY (`patient_id`) REFERENCES `patient` (`id`) ON DELETE CASCADE;

